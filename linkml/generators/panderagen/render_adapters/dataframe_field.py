from linkml.generators.oocodegen import OOField


class DataframeField(OOField):
    """Serves as an adapter between the template that renders the form of the
    dataframe schema fields and the LinkML model and schema view.

    Currently a thin wrapper around OOField
    until the dataframe requirements are fully understood.
    """

    def __init__(
        self,
        name,
        range=None,
        default_value=None,
        annotations=None,
        source_slot=None,
        inline_id_column_name: str = None,
        inline_id_other_name: str = None,
        reference_class: str = None,
        inline_form: str = None,
    ):
        super().__init__(name, range, default_value, annotations, source_slot)
        self.inline_id_column_name = inline_id_column_name
        self.inline_id_other_name = inline_id_other_name
        self.reference_class = reference_class
        self.inline_form = inline_form

    def maximum_value(self):
        return self.source_slot.maximum_value

    def minimum_value(self):
        return self.source_slot.minimum_value

    def pattern(self):
        return self.source_slot.pattern

    def minimum_cardinality(self):
        return self.source_slot.minimum_cardinality

    def maximum_cardinality(self):
        return self.source_slot.maximum_cardinality

    def permissible_values(self):
        return self.source_slot.annotations._get("permissible_values", [])

    def required(self):
        return self.source_slot.required

    def identifier(self):
        return self.source_slot.identifier

    def description(self):
        return self.source_slot.description
