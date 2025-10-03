import polars as pl


class SimpleDictLoader:
    def __init__(
        self,
        struct_schema,
        id_col="id",
        other_col="other",
        id_dtype=pl.String,
        other_dtype=pl.Int64,
        nested_tx=lambda x: x,
    ):
        self.struct_schema = struct_schema
        self.id_col: str = id_col
        self.other_col: str = other_col
        self.id_dtype = id_dtype
        self.other_dtype = other_dtype
        self.nested_tx = nested_tx
        self.polars_schema_keys = set(self.struct_schema.keys())

    def tx_core(self, linkml_simple_dict):
        """core simple dict to list of dicts logic"""
        for id_value, range_value in linkml_simple_dict.items():
            if isinstance(range_value, dict) and (set(range_value.keys()) <= self.polars_schema_keys):
                yield {
                    **{k: None for k in self.polars_schema_keys},  # make sure all values present
                    **self.nested_tx(range_value),  # copy value
                    self.id_col: id_value,  # make sure optional key is present
                }
            else:
                yield {
                    **{k: None for k in self.polars_schema_keys},  # make sure all values present
                    self.id_col: id_value,
                    self.other_col: range_value,
                }

    # simple dict handling
    def tx(self, sd):
        """simple dict to list of dicts"""
        return self.tx_core(sd)

    def load(self, source_col):
        return pl.col(source_col).map_elements(self.tx, return_dtype=pl.List(pl.Struct(self.struct_schema)))

    def load_df(self, df, source_col):
        return df.with_columns(self.load(source_col))
