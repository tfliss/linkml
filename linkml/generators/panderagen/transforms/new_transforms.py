import polars as pl


class DictToStructTransformer:
    def __init__(self, id_col="id", other_col="other", id_dtype=pl.String, other_dtype=pl.Int64, nested_tx=lambda x: x):
        self.id_col = id_col
        self.other_col = other_col
        self.id_dtype = id_dtype
        self.other_dtype = other_dtype
        self.nested_tx = nested_tx

    @staticmethod
    def tx_core(sd, id_col="id", other_col="other", nested_tx=lambda x: x):
        """core simple dict to list of dicts logic"""
        return [{id_col: k, other_col: nested_tx(v)} for k, v in sd.items()]

    # simple dict handling
    def tx(self, sd):
        """simple dict to list of dicts"""
        return self.tx_core(sd, self.id_col, self.other_col, self.nested_tx)

    def load(self, source_col):
        result = pl.col(source_col).map_elements(
            self.tx, return_dtype=pl.List(pl.Struct({self.id_col: self.id_dtype, self.other_col: self.other_dtype}))
        )
        return result

    def load_df(self, df, source_col):
        result = df.with_columns(self.load(source_col))
        return result


class CollectionToStructTransformer:
    def __init__(self, struct_schema, id_col="id", nested_tx=lambda x: x):
        self.struct_schema = struct_schema
        self.id_col = id_col
        self.nested_tx = nested_tx

    @staticmethod
    def tx_core(collection_dict, id_col="id", nested_tx=lambda x: x):
        """core collection to structs logic"""
        result = []
        for k, v in collection_dict.items():
            struct_dict = {**nested_tx(v), id_col: k}  # Collection key overwrites nested value
            result.append(struct_dict)
        return result

    def tx(self, collection_dict):
        """collection_to_structs"""
        return self.tx_core(collection_dict, self.id_col, self.nested_tx)

    def load(self, source_col):
        result = pl.col(source_col).map_elements(self.tx, return_dtype=pl.List(self.struct_schema))
        return result

    def load_df(self, df, source_col):
        result = df.with_columns(self.load(source_col))
        return result
