import polars as pl

from .model_transform import ModelTransform


class CollectionDictModelTransform(ModelTransform):
    """This class assists in converting a LinkML 'collection dict' inline column
    into a form that is better for representing in a PolaRS dataframe and
    validating with a Pandera model.
    """

    def __init__(self, nested_cls, polars_schema, polars_schema_dict):
        self.nested_cls = nested_cls
        self.polars_schema = polars_schema
        """Polars Schema representing the nested class without the collection dict"""

        self.polars_schema_dict = polars_schema_dict
        """A polars schema representing a collection dict column"""

        self.id_col = nested_cls.get_id_column_name()
        """The ID column in the sense of a LinkML inline collection dict"""

    def transform(self, linkml_collection_dict):
        """Converts a collection dict nested column to a list of dicts.
        { 'A': {...}, 'B': {...}, ... } -> [{'id': 'A', ...}, {'id': 'B', ...}, ...]
        """
        return self._collection_dict_to_list_of_structs(linkml_collection_dict)

    def _collection_dict_to_list_of_structs(self, linkml_collection_dict):
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
                v[self.id_col] = k
            arr.append(v)
        return arr

    def prepare_series(self, lf: pl.LazyFrame, column_name: str) -> pl.Series:
        """Returns just the collection dict column transformed to an inlined list form

        note that this method uses collect and iter_rows so is very inefficient
        """
        one_column_df = lf.select(pl.col(column_name)).collect()

        list_of_structs = []
        for [e] in one_column_df.iter_rows():
            transformed = self.transform(e)
            if len(transformed) > 0:
                list_of_structs.append(transformed)

        if len(list_of_structs) == 0:
            list_of_structs = None

        return pl.Series(column_name, list_of_structs, dtype=self.polars_schema_dict)

    def prepare_dataframe(self, data, column_name: str):
        """Returns just the collection dict column transformed to an inlined list form"""
        # list_of_structs = data.lazyframe.select(pl.col(column_name)).collect().to_dicts().get(column_name)

        return pl.DataFrame(self.prepare_series(data.lazyframe, column_name))

    def explode_unnest_dataframe(self, df, column_name):
        """Filter, explode and unnest for collection dict."""
        return df.lazy().filter(pl.col(column_name).list.len() > 0).explode(column_name).unnest(column_name).collect()
