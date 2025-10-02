import polars as pl

from .model_transform import ModelTransform


class SimpleDictModelTransform(ModelTransform):
    """This class assists in converting a LinkML 'simple dict' inline column
    into a form that is better for representing in a PolaRS dataframe and
    validating with a Pandera model.
    """

    def __init__(self, id_col, other_col, polars_schema, polars_schema_dict):
        self.id_col = id_col
        self.other_col = other_col
        self.polars_schema = polars_schema
        """A polars schema representing a simple dict column"""

        self.polars_schema_keys = set(self.polars_schema.keys())

        self.polars_struct = polars_schema_dict  # self._build_polars_struct()
        """A pl.Struct representing the schema of the other range."""

        self.nested_tx = lambda x: x
        """No-op may be used in the future to handle nested inlined forms"""

    def transform(self, linkml_simple_dict):
        """Converts a simple dict nested column to a list of dicts.
        { 'A': 1, 'B': 2, ... } -> [{'id': 'other': 1}, {'id': 'B', 'other': 2}, ...]
        """
        return list(self._simple_dict_to_list_of_structs(linkml_simple_dict))

    def _simple_dict_to_list_of_structs(self, linkml_simple_dict):
        """Converts a simple dict nested column to a list of dicts.
        { 'A': 1, 'B': 2, ... } -> [{'id': 'other': 1}, {'id': 'B', 'other': 2}, ...]

        An inefficient conversion (relative to native PolaRS operations)
        from a simple dict form to a dataframe struct column.

        e : dict
            e is a single row entry in a dataframe column (one cell), which itself is a dict.
            The value entries of e may also be dicts.
        """
        for id_value, range_value in linkml_simple_dict.items():
            if isinstance(range_value, dict) and (set(range_value.keys()) <= self.polars_schema_keys):
                yield {
                    **{k: None for k in self.polars_schema_keys},  # make sure all values present
                    **self.nested_tx(range_value),  # copy value
                    self.id_col: id_value,  # make sure optional key is present
                }
            else:
                yield {self.id_col: id_value, self.other_col: range_value}

    def list_dtype(self):
        return pl.List(self.polars_struct)

    def explode_unnest_dataframe(self, df, column_name):
        """Explode and unnest for simple dict."""
        # fmt: off
        return (
            df.lazy()
            .explode(column_name)
            .unnest(column_name)
            .collect()
        )
