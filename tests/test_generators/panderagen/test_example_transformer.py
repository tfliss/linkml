import polars as pl

sds = [{"A": 1, "B": 2}, {"C": 3, "D": 4}, {"E": 5, "F": 6, "G": 7}]
sds2 = [{"A": {"b": 1, "c": 2}, "B": {"b": 3, "c": 4}}, {"C": {"b": 5, "c": 6}, "D": {"b": 7, "c": 8}}]
sds3 = [
    {"X": {"A": {"b": 1, "c": 2}, "B": {"b": 3, "c": 4}}, "Y": {"C": {"b": 5, "c": 6}, "D": {"b": 7, "c": 8}}},
    {"Z": {"E": {"b": 9, "c": 10}, "F": {"b": 11, "c": 12}}},
]


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

    def load(self, df, source_col):
        return df.with_columns(
            pl.col(source_col).map_elements(
                self.tx, return_dtype=pl.List(pl.Struct({self.id_col: self.id_dtype, self.other_col: self.other_dtype}))
            )
        )


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

    def load(self, df, source_col):
        return df.with_columns(pl.col(source_col).map_elements(self.tx, return_dtype=pl.List(self.struct_schema)))


def test_one():
    df = pl.DataFrame({"t": sds2}, schema={"t": pl.Object})
    print(df)

    tx = CollectionToStructTransformer(
        struct_schema=pl.Struct({"id": pl.String, "b": pl.String, "c": pl.Int64}), id_col="id"
    )
    tx_df = tx.load(df, "t")
    print(tx_df)
    assert tx_df.schema["t"] == pl.List(pl.Struct({"id": pl.String, "b": pl.String, "c": pl.Int64, "id": pl.String}))


def test_two():
    df = pl.DataFrame({"t": sds}, schema={"t": pl.Object})
    print(df)

    tx = DictToStructTransformer(id_col="id", other_col="other")
    tx_df = tx.load(df, "t")

    print(tx_df)
    assert tx_df.schema["t"] == pl.List(pl.Struct({"id": pl.String, "other": pl.Int64}))


def test_three():
    nested_dtype = pl.Struct({"id": pl.String, "b": pl.Int64, "c": pl.Int64})
    other_dtype = pl.List(nested_dtype)
    t_dtype = pl.List(pl.Struct({"id": pl.String, "other": other_dtype}))
    orig_t_dtype = pl.Object
    df_dict = {"t": orig_t_dtype}

    df = pl.DataFrame({"t": sds3}, schema=df_dict)
    print(df)

    nested_tx = lambda collection_dict: CollectionToStructTransformer.tx_core(collection_dict, "id")
    tx = DictToStructTransformer(id_col="id", other_col="other", other_dtype=other_dtype, nested_tx=nested_tx)
    tx_df = tx.load(df, "t")

    print(tx_df)
    assert tx_df.schema["t"] == t_dtype
