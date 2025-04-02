import logging

from linkml.generators.oocodegen import OOField

logger = logging.getLogger(__file__)


class SlotGeneratorMixin:
    LINKML_ANY_CURIE = "linkml:Any"
    ANY_RANGE_STRING = "Object"
    CLASS_RANGE_STRING = "Struct"
    ENUM_RANGE_STRING = "Enum"

    # to be implemented by the class
    def make_multivalued(self, range: str):
        raise NotImplementedError("please implement make multivalued in the class")

    def handle_none_slot(self, slot, range: str) -> str:
        range = self.schema.default_range  # need to figure this out, set at the beginning?
        if range is None:
            range = "str"

        return range

    def handle_class_slot(self, slot, range: str) -> str:
        range_info = self.schemaview.all_classes().get(range)

        if range_info["class_uri"] == SlotGeneratorMixin.LINKML_ANY_CURIE:
            range = SlotGeneratorMixin.ANY_RANGE_STRING
        elif slot.inlined:
            range = self.handle_inlined_as_simple_dict_class_slot(slot, range)
        elif slot.inlined_as_list:
            range = self.handle_inlined_as_list_class_slot(slot, range)
        else:
            range = self.handle_non_inlined_class_slot(slot, range)

        return range

    def handle_inlined_as_list_class_slot(self, slot, range) -> str:
        slot.annotations["reference_class"] = self.get_class_name(range)

        range = SlotGeneratorMixin.CLASS_RANGE_STRING

        return range

    def handle_inlined_as_simple_dict_class_slot(self, slot, range: str) -> str:
        raise NotImplementedError("Inlining as a simple dict not supported by PanderaGen")

    def handle_non_inlined_class_slot(self, slot, range: str) -> str:
        return f"ID_TYPES['{self.get_class_name(range)}']"

    def handle_type_slot(self, slot, range: str) -> str:
        del slot  # unused for now

        t = self.schemaview.all_types().get(range)
        range = self.map_type(t)

        return range

    def handle_enum_slot(self, slot, range: str) -> str:
        enum_definition = self.schemaview.all_enums().get(range)
        range = SlotGeneratorMixin.ENUM_RANGE_STRING
        slot.annotations["permissible_values"] = self.get_enum_permissible_values(enum_definition)

        return range

    # These handlers might need to go in the class proper
    def handle_multivalued_slot(self, slot, range: str) -> str:
        if slot.multivalued:
            if slot.inlined_as_list and range != SlotGeneratorMixin.CLASS_RANGE_STRING:
                range = self.make_multivalued(range)

        return range

    def handle_slot(self, cn: str, sn: str):
        safe_sn = self.get_slot_name(sn)
        slot = self.schemaview.induced_slot(sn, cn)
        range = slot.range

        if slot.alias is not None:
            safe_sn = self.get_slot_name(slot.alias)

        if range is None:
            range = self.handle_none_slot(slot, range)
        elif range in self.schemaview.all_classes():
            range = self.handle_class_slot(slot, range)
        elif range in self.schemaview.all_types():
            range = self.handle_type_slot(slot, range)
        elif range in self.schemaview.all_enums():
            range = self.handle_enum_slot(slot, range)
        else:
            raise Exception(f"Unknown range {range}")

        range = self.handle_multivalued_slot(slot, range)

        return OOField(
            name=safe_sn,
            source_slot=slot,
            range=range,
        )
