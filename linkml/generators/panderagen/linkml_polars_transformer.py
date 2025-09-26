import logging

import polars as pl

logger = logging.getLogger(__name__)


class LinkmlPolarsTransformer:
    def __init__(self, polars_schema_dict: dict[str, pl.DataType]):
        self.polars_schema_dict = polars_schema_dict
        """dtype parameter for table destination schema, not columns."""

    def _collection_dict_to_list_of_structs(self, linkml_collection_dict, id_col: str) -> list:
        """Converts a collection dict nested column to a list of dicts.
        { 'A': {...}, 'B': {...}, ... } -> [{'id': 'A', ...}, {'id': 'B', ...}, ...]

        An inefficient conversion (relative to native PolaRS operations)
        from a collection dict form to a dataframe struct column.

        linkml_collection_dict : dict
            A single row entry in a dataframe column (one cell), which itself is a dict.
            The value entries are dicts that get the key added as an id field.
        """
        arr = []

        for k, v in linkml_collection_dict.items():
            if k not in v:
                v[id_col] = k
            arr.append(v)

        return arr

    def _simple_dict_to_list_of_structs(self, linkml_simple_dict, col_name, id_col, other_col, range_schema_dict):
        """Converts a simple dict nested column to a list of dicts.
        { 'A': 1, 'B': 2, ... } -> [{'id': 'other': 1}, {'id': 'B', 'other': 2}, ...]

        An inefficient conversion (relative to native PolaRS operations)
        from a simple dict form to a dataframe struct column.

        e : dict
            e is a single row entry in a dataframe column (one cell), which itself is a dict.
            The value entries of e may also be dicts.
        """
        arr = []

        polars_schema_keys = set(range_schema_dict.keys())

        #
        # OK this logic needs to be updated now that we know the actual
        # simple dict schema
        #
        for id_value, range_value in linkml_simple_dict.items():
            range_dict = self.compute_range_dict(id_value, range_value, id_col, other_col, polars_schema_keys)
            if len(range_dict.keys()) > 0:
                arr.append(range_dict)

        return arr

    def compute_range_dict(self, id_value, range_value, id_col, other_col, polars_schema_keys) -> dict:
        if isinstance(range_value, dict) and (set(range_value.keys()) <= polars_schema_keys):
            range_dict = range_value
            range_dict[id_col] = id_value
            for column_key in polars_schema_keys:
                if column_key not in range_dict:
                    range_dict[column_key] = None  # might not need this w/ specified schema?
        else:
            # this can be returned as two separate series, avoid packing unpacking in python
            # nested transform gets applied to range value.
            range_dict = {id_col: id_value, other_col: range_value}

        return range_dict

    def transform_simple_dict_list():
        pass

    def transform_simple_dict_entry():
        pass

    def transform_collection_dict(self, df, column_name, reference_cls=None) -> pl.Series:
        """polars_schema_dict is the new schema"""

        one_column_df = df.lazy().select(pl.col(column_name)).collect()

        list_of_structs = []

        for [e] in one_column_df.iter_rows():
            transformed = self._collection_dict_to_list_of_structs(e, "id")
            if len(transformed) > 0:
                list_of_structs.append(transformed)

        #
        # TODO: need to recursively transform here
        #

        if len(list_of_structs) == 0:
            list_of_structs = None

        return pl.Series(column_name, list_of_structs, dtype=pl.List(reference_cls))

    def transform_simple_dict(
        self, df, column_name, id_col, other_col, range_schema_dict, range_schema_struct, nested_fn
    ):
        one_column_df = df.lazy().select(pl.col(column_name)).collect()

        list_of_structs = []

        # make this a function with a generator
        for [e] in one_column_df.iter_rows():
            transformed = self._simple_dict_to_list_of_structs(e, column_name, id_col, other_col, range_schema_dict)
            if len(transformed) > 0:
                list_of_structs.append(transformed)

        #
        # TODO: need to recursively transform here
        #

        if len(list_of_structs) == 0:
            list_of_structs = None

        # check if list of structs can be a generator

        if nested_fn is None:
            return pl.Series(column_name, list_of_structs, dtype=pl.List(range_schema_struct))
        elif column_name == "double_nested_simple_dict_column":
            d = pl.DataFrame(list_of_structs, orient="row")
            return d.to_series()
        elif column_name == "inlined_nested_simple_dict_column":
            #
            # What I need to do here is also record the nested form of other
            # and add a special function to handle that.
            #
            # it's going to be something like check the inline form
            # and then cls.transform_xxx_dict(nested_df, some col, etc.)
            #
            # nested_df = pl.DataFrame(pl.Series)
            # logger.info(nested_df)

            # new schema is: Schema({'inlined_nested_simple_dict_column': List(Struct({'id': String, 'nested_list': List(Struct({'id': String, 'x': Int64, 'y': Int64}))}))})
            # which is now a list of dicts at top level and collection dict inside.

            return pl.DataFrame(list_of_structs).to_series()

    def transform_list_dict(self, df, column_name, transform_fn, target_dtype) -> pl.Series:
        return df.select(
            pl.col(column_name).list.eval(pl.map_elements(transform_fn, return_dtype=target_dtype))
        ).to_series()
