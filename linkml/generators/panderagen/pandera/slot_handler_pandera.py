import logging
from typing import TYPE_CHECKING

from linkml.utils.helpers import get_range_associated_slots

from ..render_adapters.dataframe_field import DataframeField
from ..slot_handler_base import SlotHandlerBase

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__file__)


class SlotHandlerPandera(SlotHandlerBase):
    """
    Prior to rendering the dataframe schema, this class provides
    and adapter between the LinkML model and schema view
    and the rendering engine.
    """

    # TODO: maybe can move this to base class
    def backing_inlined_form(self, inlined_form: str) -> str:
        loaded_form = {
            SlotHandlerBase.FORM_INLINED_SIMPLE_DICT: SlotHandlerBase.FORM_INLINED_LIST_DICT,
            SlotHandlerBase.FORM_INLINED_COLLECTION_DICT: SlotHandlerBase.FORM_INLINED_LIST_DICT,
        }

        if self.generator.backing_form in ["serialization"]:
            return inlined_form
        elif self.generator.backing_form in ["loaded", "transform"]:
            return loaded_form.get(inlined_form, inlined_form)

        logger.warning(f"Unknown backing form: {self.generator.backing_form}")
        return inlined_form

    # constants used to render the schema
    # these will be moved to a dialect-specific place
    ANY_RANGE_STRING = "Object"
    CLASS_RANGE_STRING = "Struct"
    SIMPLE_DICT_RANGE_STRING = "Object"
    ENUM_RANGE_STRING = "Enum"

    # When nested inlining is done, the Pandera validator needs a specific range
    INLINED_FORM_RANGE_PANDERA = {
        SlotHandlerBase.FORM_INLINED_SIMPLE_DICT: SIMPLE_DICT_RANGE_STRING,
        SlotHandlerBase.FORM_INLINED_LIST_DICT: CLASS_RANGE_STRING,
        SlotHandlerBase.FORM_INLINED_COLLECTION_DICT: ANY_RANGE_STRING,
        SlotHandlerBase.FORM_INLINED_DICT: CLASS_RANGE_STRING,
        SlotHandlerBase.FORM_ERROR: None,
    }

    def handle_none_slot(self, slot, field: DataframeField) -> None:
        del slot  # unused for now
        range = self.generator.schema.default_range  # need to figure this out, set at the beginning?

        if range is None:
            range = "str"

        field.range = range

    def handle_class_slot(self, slot, field) -> None:
        range = slot.range
        range_info = self.generator.schemaview.all_classes().get(range)

        field.reference_class = self.generator.get_class_name(range)

        if range_info and range_info["class_uri"] == SlotHandlerBase.LINKML_ANY_CURIE:
            range = self.__class__.ANY_RANGE_STRING
        else:
            inlined_form = self.backing_inlined_form(self.calculate_inlined_form(slot))
            field.inline_form = inlined_form

            if inlined_form == SlotHandlerBase.FORM_INLINED_COLLECTION_DICT:
                logger.warning(
                    f"Slot {slot.name} uses inlined dictionary form,"
                    "which may be less efficient than inlined as list form with the current implementation."
                )
                range = SlotHandlerPandera.INLINED_FORM_RANGE_PANDERA[inlined_form]
            elif inlined_form == SlotHandlerBase.FORM_INLINED_SIMPLE_DICT:
                logger.warning(
                    f"Slot {slot.name} uses inlined simple dictionary form. Support is incomplete "
                    "and performance is less efficient than inlined as list form with the current implementation."
                )
                range = SlotHandlerPandera.INLINED_FORM_RANGE_PANDERA[inlined_form]
                self.set_simple_dict_inline_details(slot, field)
            elif inlined_form == SlotHandlerBase.FORM_INLINED_DICT:
                range = SlotHandlerPandera.INLINED_FORM_RANGE_PANDERA[inlined_form]
            elif inlined_form == SlotHandlerBase.FORM_MULTIVALUED_FOREIGN_KEY:
                range = self.generator.make_multivalued(f"ID_TYPES['{self.generator.get_class_name(range)}']")
            elif inlined_form == SlotHandlerBase.FORM_FOREIGN_KEY:
                range = f"ID_TYPES['{self.generator.get_class_name(range)}']"
            else:
                range = SlotHandlerPandera.INLINED_FORM_RANGE_PANDERA[inlined_form]

                if inlined_form in [SlotHandlerBase.FORM_INLINED_LIST_DICT]:
                    range = self.generator.make_multivalued(range)

        field.range = range

    def set_simple_dict_inline_details(self, slot, field) -> None:
        """Extra metadata is to help with the simple dict case"""
        (range_id_slot, range_simple_dict_value_slot, _) = get_range_associated_slots(  # range_required_slots,
            self.generator.schemaview, slot.range
        )

        field.inline_id_column_name = range_id_slot.name
        field.inline_other_column_name = range_simple_dict_value_slot.name

    def handle_non_inlined_class_slot(self, slot, field) -> None:
        """non-inlined class slots have been temporarily removed but this will be needed to support them"""
        # TODO: resolve this earlier
        range = slot.range
        field.range = f"ID_TYPES['{self.generator.get_class_name(range)}']"

    def handle_type_slot(self, slot, field) -> None:
        range = slot.range

        t = self.generator.schemaview.all_types().get(range)
        range = self.generator.map_type(t)

        if self.is_multivalued(slot):
            range = self.handle_multivalued_slot(slot, range)

        field.range = range

    def handle_enum_slot(self, slot, field) -> None:
        """Returns the name of the generated Python variable containing the enum"""
        range = slot.range
        enum_definition = self.get_enum_definition(range)
        range = self.__class__.ENUM_RANGE_STRING
        field.permissible_values = self.generator.enum_handler.get_enum_permissible_values(enum_definition)

        if self.is_multivalued(slot):
            range = self.handle_multivalued_slot(slot, range)

        field.range = range
