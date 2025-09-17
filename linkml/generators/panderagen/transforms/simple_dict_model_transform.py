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

        # schema_keys = list(polars_schema.keys()) # needs improvement

        # if len(schema_keys) >= 2:
        #     self.id_col = schema_keys[0]
        #     self.other_col = schema_keys[1]

        # self.id_col_type = None

        # TODO: might not need these any more
        # self.other_col_type = None
        self.polars_struct = polars_schema_dict  # self._build_polars_struct()
        """A pl.Struct representing the schema of the other range."""

    # def _build_polars_struct_simple(self):
    #     """Handles the two column (id, other) form of the simple dict"""
    #     self.id_col_type = self.polars_schema.columns[self.id_col].dtype.type
    #     self.other_col_type = self.polars_schema.columns[self.other_col].dtype.type

    #     return pl.Struct({self.id_col: self.id_col_type, self.other_col: self.other_col_type})

    # def _build_polars_struct_complex(self):
    #     """Handles the non-two-column simple dict cases."""
    #     struct_items = {}
    #     for k, v in self.polars_schema.columns.items():
    #         if v.dtype.type == pl.Object:
    #             v.dtype.type = pl.Struct
    #         else:
    #             struct_items[k] = v.dtype.type
    #     return pl.Struct(struct_items)

    # def _build_polars_struct(self):
    #     if len(self.polars_schema.columns.keys()) == 2:
    #         return self._build_polars_struct_simple()
    #     else:
    #         return self._build_polars_struct_complex()

    def transform(self, linkml_simple_dict):
        """Converts a simple dict nested column to a list of dicts.
        { 'A': 1, 'B': 2, ... } -> [{'id': 'other': 1}, {'id': 'B', 'other': 2}, ...]
        """
        return self._simple_dict_to_list_of_structs(linkml_simple_dict)

    def _simple_dict_to_list_of_structs(self, linkml_simple_dict):
        """Converts a simple dict nested column to a list of dicts.
        { 'A': 1, 'B': 2, ... } -> [{'id': 'other': 1}, {'id': 'B', 'other': 2}, ...]

        An inefficient conversion (relative to native PolaRS operations)
        from a simple dict form to a dataframe struct column.

        e : dict
            e is a single row entry in a dataframe column (one cell), which itself is a dict.
            The value entries of e may also be dicts.
        """
        arr = []

        polars_schema_keys = set(self.polars_schema.keys())

        #
        # OK this logic needs to be updated now that we know the actual
        # simple dict schema
        #
        for id_value, range_value in linkml_simple_dict.items():
            if isinstance(range_value, dict) and (set(range_value.keys()) <= polars_schema_keys):
                range_dict = range_value
                range_dict[self.id_col] = id_value
                for column_key in polars_schema_keys:
                    if column_key not in range_dict:
                        range_dict[column_key] = None
            else:
                range_dict = {self.id_col: id_value, self.other_col: range_value}

            if len(range_dict.keys()) > 0:
                arr.append(range_dict)

        return arr

    def list_dtype(self):
        return pl.List(self.polars_struct)

    def explode_unnest_dataframe(self, df, column_name):
        """Explode and unnest for simple dict."""
        return df.lazy().explode(column_name).unnest(column_name).collect()
