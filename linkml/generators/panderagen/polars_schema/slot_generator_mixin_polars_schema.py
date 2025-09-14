import logging

from linkml.utils.helpers import get_range_associated_slots

from ..slot_generator_mixin_base import SlotGeneratorMixinBase

logger = logging.getLogger(__file__)


class SlotGeneratorMixinPolarsSchema(SlotGeneratorMixinBase):
    """
    Prior to rendering the dataframe schema, this class provides
    and adapter between the LinkML model and schema view
    and the rendering engine.
    """

    # constants used to render the schema
    # these will be moved to a dialect-specific place
    ANY_RANGE_STRING = "pl.Object"
    CLASS_RANGE_STRING = "pl.Struct"
    SIMPLE_DICT_RANGE_STRING = "pl.Struct"
    ENUM_RANGE_STRING = "pl.Enum"

    def handle_none_slot(self, slot) -> str:
        range = self.schema.default_range  # need to figure this out, set at the beginning?
        if range is None:
            range = "str"

        return range

    def handle_class_slot(self, slot, range: str) -> str:
        range_info = self.schemaview.all_classes().get(range)

        if range_info["class_uri"] == SlotGeneratorMixinBase.LINKML_ANY_CURIE:
            range = self.__class__.ANY_RANGE_STRING  # TODO: update this
        else:
            inlined_form = self.calculate_inlined_form(slot)

            #
            # Todo get rid of inline_form it's not used
            #
            slot.annotations["inline_form"] = inlined_form

            if inlined_form == SlotGeneratorMixinBase.FORM_MULTIVALUED_FOREIGN_KEY:
                range = self.make_multivalued(f"{self.range_id_type(slot)}")
            elif inlined_form == SlotGeneratorMixinBase.FORM_FOREIGN_KEY:
                range = self.range_id_type(slot)  # TODO: make this a get id function
                print(range)
            elif inlined_form in (SlotGeneratorMixinBase.FORM_INLINED_LIST_DICT):
                range = self.get_class_name(range)
                range = self.make_multivalued(f"{range}Struct")
            else:
                range = self.get_class_name(range)
                range = f"{range}Struct"

        return range

    def set_simple_dict_inline_details_annotation(self, slot):
        """Extra metadata is to help with the simple dict case"""
        (range_id_slot, range_simple_dict_value_slot, _) = get_range_associated_slots(  # range_required_slots,
            self.schemaview, slot.range
        )

        simple_dict_id = range_id_slot.name
        other_slot = range_simple_dict_value_slot.name
        slot.annotations["inline_details"] = {"id": simple_dict_id, "other": other_slot}

    def handle_non_inlined_class_slot(self, slot, range: str) -> str:
        """non-inlined class slots have been temporarily removed but this will be needed to support them"""
        return f"ID_TYPES['{self.get_class_name(range)}']"

    def handle_type_slot(self, slot, range: str) -> str:
        del slot  # unused for now

        t = self.schemaview.all_types().get(range)
        range = self.map_type(t)

        return range

    def handle_enum_slot(self, slot, range: str) -> str:
        """Returns the name of the generated Python variable containing the enum"""
        enum_definition = self.get_enum_definition(range)
        enum_name = self.get_enum_name(enum_definition.name)

        return enum_name
