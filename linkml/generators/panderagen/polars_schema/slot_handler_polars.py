import logging

from linkml.utils.helpers import get_range_associated_slots

from ..slot_handler_base import SlotHandlerBase

logger = logging.getLogger(__file__)


class SlotHandlerPolars(SlotHandlerBase):
    """
    Prior to rendering the dataframe schema, this class provides
    and adapter between the LinkML model and schema view
    and the rendering engine.
    """

    def backing_inlined_form(self, inlined_form: str) -> str:
        loaded_form = {
            SlotHandlerBase.FORM_INLINED_SIMPLE_DICT: SlotHandlerBase.FORM_INLINED_SIMPLE_DICT,
            SlotHandlerBase.FORM_INLINED_COLLECTION_DICT: SlotHandlerBase.FORM_INLINED_LIST_DICT,
        }

        if self.generator.backing_form in "serialization":
            return inlined_form
        elif self.generator.backing_form in ["loaded", "transform"]:
            return loaded_form.get(inlined_form, inlined_form)

        logger.warning(f"Unknown backing form: {self.generator.backing_form}")
        return inlined_form

    # constants used to render the schema
    # these will be moved to a dialect-specific place
    ANY_RANGE_STRING = "pl.Object"
    CLASS_RANGE_STRING = "pl.Struct"
    SIMPLE_DICT_RANGE_STRING = "pl.Struct"
    ENUM_RANGE_STRING = "pl.Enum"

    def handle_none_slot(self, slot, field) -> None:
        del slot
        range = self.generator.schema.default_range  # need to figure this out, set at the beginning?
        if range is None:
            range = "str"

        field.range = range

    def handle_class_slot(self, slot, field) -> None:
        range = slot.range
        range_info = self.generator.schemaview.all_classes().get(range)
        field.reference_class = self.generator.get_class_name(range)

        if range_info["class_uri"] == SlotHandlerBase.LINKML_ANY_CURIE:
            range = self.__class__.ANY_RANGE_STRING  # TODO: update this
        else:
            inlined_form = self.backing_inlined_form(self.calculate_inlined_form(slot))
            field.inline_form = inlined_form

            if inlined_form == SlotHandlerBase.FORM_MULTIVALUED_FOREIGN_KEY:
                range = self.generator.make_multivalued(f"{self.range_id_type(slot)}")
            elif inlined_form == SlotHandlerBase.FORM_FOREIGN_KEY:
                range = self.range_id_type(slot)  # TODO: make this a get id function
                print(range)
            elif inlined_form in (SlotHandlerBase.FORM_INLINED_LIST_DICT):
                range = self.generator.get_class_name(range)
                range = self.generator.make_multivalued(f"{range}Struct")
            elif inlined_form == SlotHandlerBase.FORM_INLINED_COLLECTION_DICT:
                range = SlotHandlerPolars.ANY_RANGE_STRING
            elif inlined_form == SlotHandlerBase.FORM_INLINED_SIMPLE_DICT:
                self.set_simple_dict_inline_details(slot, field)
                range = SlotHandlerPolars.ANY_RANGE_STRING
            else:
                range = self.generator.get_class_name(range)
                range = f"{range}Struct"

        field.range = range

    def set_simple_dict_inline_details(self, slot, field) -> None:
        """Extra metadata is to help with the simple dict case"""
        (range_id_slot, range_simple_dict_value_slot, _) = get_range_associated_slots(  # range_required_slots,
            self.generator.schemaview, slot.range
        )

        field.inline_id_column_name = range_id_slot.name
        field.inline_other_column_name = range_simple_dict_value_slot.name

        other_range = range_simple_dict_value_slot.range

        if other_range in self.generator.schemaview.all_classes():
            field.inline_other_range = self.generator.get_class_name(other_range)

    def handle_non_inlined_class_slot(self, slot, field) -> None:
        """non-inlined class slots have been temporarily removed but this will be needed to support them"""
        range = slot.range
        field.range = f"ID_TYPES['{self.generator.get_class_name(range)}']"

    def handle_type_slot(self, slot, field) -> None:
        t = self.generator.schemaview.all_types().get(slot.range)
        range = self.generator.map_type(t)

        if self.is_multivalued(slot):
            range = self.handle_multivalued_slot(slot, range)

        field.range = range

    def handle_enum_slot(self, slot, field) -> None:
        """Returns the name of the generated Python variable containing the enum"""
        enum_definition = self.get_enum_definition(slot.range)
        enum_name = self.generator.get_enum_name(enum_definition.name)

        if self.is_multivalued(slot):
            enum_name = self.handle_multivalued_slot(slot, enum_name)

        field.range = enum_name
