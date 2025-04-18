import logging

from linkml_runtime.linkml_model.meta import SlotDefinition

from linkml.generators.oocodegen import OOField
from linkml.utils.helpers import get_range_associated_slots

logger = logging.getLogger(__file__)


class SlotGeneratorMixin:
    LINKML_ANY_CURIE = "linkml:Any"
    ANY_RANGE_STRING = "Object"
    CLASS_RANGE_STRING = "Struct"
    SIMPLE_DICT_RANGE_STRING = "Object"
    ENUM_RANGE_STRING = "Enum"
    FORM_INLINED_DICT = "inlined_dict"
    FORM_INLINED_LIST_DICT = "inlined_list_dict"
    FORM_INLINED_SIMPLE_DICT = "simple_dict"

    def is_multivalued(self, slot):
        return "multivalued" in slot and slot.multivalued

    def calculate_simple_dict(self, slot: SlotDefinition):
        """slot is the container for the simple dict slot"""

        (_, range_simple_dict_value_slot, _) = get_range_associated_slots(self.schemaview, slot.range)

        return range_simple_dict_value_slot

    def handle_none_slot(self, slot, range: str) -> str:
        range = self.schema.default_range  # need to figure this out, set at the beginning?
        if range is None:
            range = "str"

        return range

    def handle_class_slot(self, slot, range: str) -> str:
        range_info = self.schemaview.all_classes().get(range)

        if range_info["class_uri"] == SlotGeneratorMixin.LINKML_ANY_CURIE:
            range = SlotGeneratorMixin.ANY_RANGE_STRING
        elif slot.inlined_as_list:
            range = self.handle_inlined_class_slot(slot, range)
        elif slot.inlined:
            if self.calculate_simple_dict(slot):
                range = self.handle_inlined_as_simple_dict_class_slot(slot, range)
            else:
                range = self.handle_inlined_class_slot(slot, range)
        else:
            range = self.handle_non_inlined_class_slot(slot, range)

        return range

    def handle_inlined_class_slot(self, slot, range) -> str:
        slot.annotations["reference_class"] = self.get_class_name(range)
        if slot.multivalued:
            slot.annotations["inline_form"] = SlotGeneratorMixin.FORM_INLINED_LIST_DICT
        else:
            slot.annotations["inline_form"] = SlotGeneratorMixin.FORM_INLINED_DICT

        range = SlotGeneratorMixin.CLASS_RANGE_STRING

        return range

    def handle_inlined_as_simple_dict_class_slot(self, slot, range: str) -> str:
        slot.annotations["reference_class"] = self.get_class_name(range)
        range = SlotGeneratorMixin.SIMPLE_DICT_RANGE_STRING  # range is getting set to object multiple times

        (range_id_slot, range_simple_dict_value_slot, _) = get_range_associated_slots(  # range_required_slots,
            self.schemaview, slot.range
        )

        simple_dict_id = range_id_slot.name
        other_slot = range_simple_dict_value_slot.name
        slot.annotations["inline_details"] = {"id": simple_dict_id, "other": other_slot}

        slot.annotations["inline_form"] = SlotGeneratorMixin.FORM_INLINED_SIMPLE_DICT

        return range

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

    def handle_multivalued_slot(self, slot, range: str) -> str:
        if slot.inlined_as_list:  # and range != SlotGeneratorMixin.CLASS_RANGE_STRING:
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
