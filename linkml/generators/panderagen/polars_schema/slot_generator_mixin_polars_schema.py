import logging

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

    def handle_none_slot(self, slot, field) -> None:
        del slot
        range = self.schema.default_range  # need to figure this out, set at the beginning?
        if range is None:
            range = "str"

        field.range = range

    def handle_class_slot(self, slot, field) -> None:
        range = slot.range
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
            elif inlined_form in (
                SlotGeneratorMixinBase.FORM_INLINED_COLLECTION_DICT,
                SlotGeneratorMixinBase.FORM_INLINED_SIMPLE_DICT,
            ):
                range = SlotGeneratorMixinPolarsSchema.ANY_RANGE_STRING
            else:
                range = self.get_class_name(range)
                range = f"{range}Struct"

        field.range = range

    def handle_non_inlined_class_slot(self, slot, field) -> None:
        """non-inlined class slots have been temporarily removed but this will be needed to support them"""
        range = slot.range
        field.range = f"ID_TYPES['{self.get_class_name(range)}']"

    def handle_type_slot(self, slot, field) -> None:
        t = self.schemaview.all_types().get(slot.range)
        range = self.map_type(t)

        if self.is_multivalued(slot):
            range = self.handle_multivalued_slot(slot, range)

        field.range = range

    def handle_enum_slot(self, slot, field) -> None:
        """Returns the name of the generated Python variable containing the enum"""
        enum_definition = self.get_enum_definition(slot.range)
        enum_name = self.get_enum_name(enum_definition.name)

        if self.is_multivalued(slot):
            enum_name = self.handle_multivalued_slot(slot, enum_name)

        field.range = enum_name
