import inspect

import narwhals as nw
import pandera.polars as pla
import polars as pl
from pandera.api.polars.types import PolarsData


class LinkmlPanderaValidator:
    @classmethod
    def _check_cardinality(cls, df, column_name, min_cardinality: int = None, max_cardinality: int = None):
        min_expr = nw.col("count") >= min_cardinality
        max_expr = nw.col("count") <= max_cardinality

        if min_cardinality is not None:
            if max_cardinality is not None:
                combined_expression = (min_expr and max_expr).all()
            else:
                combined_expression = min_expr.all()
        else:
            combined_expression = max_expr.all()

        df = (
            nw.from_native(df)
            .group_by(column_name)
            .agg(nw.col(column_name).count().alias("count"))
            .select(combined_expression)
            .to_native()
        )

        return df

    @classmethod
    def _simple_dict_fields(cls, column_name, pandera_model: pla.DataFrameModel):
        details = cls._INLINE_DETAILS[column_name]  # <-- THESE ARE GOING ON THE OUTER CLASS

        return (details["id"], details["other"])

    @classmethod
    def _check_simple_dict(cls, pandera_model: pla.DataFrameModel, data: PolarsData):
        """ """
        column_name = data.key
        #
        # GET MORE INFORMATION HERE:
        # simple_dict_schema_info = cls.get_nested_range(column_name).to_schema().columns
        #
        polars_schema = cls.get_nested_range(column_name).to_schema()
        (id_column, other_column) = cls._simple_dict_fields(column_name, pandera_model)

        # fmt: off
        df = (
            data.lazyframe
            .select(
                cls.simple_dict_to_list_of_structs_expr(
                    column_name,
                    id_column,
                    other_column,
                    polars_schema
                )
            )
            .filter(pl.col(column_name).list.len() > 0) # see: https://github.com/pola-rs/polars/issues/14381
            .explode(column_name)
            .unnest(column_name)
        )
        # fmt: on

        try:
            nested_cls = cls.get_nested_range(column_name)
            nested_cls.validate(df, lazy=True)
        except Exception:
            return data.lazyframe.select(pl.lit(False))

        return data.lazyframe.select(pl.lit(True))

    @classmethod
    def _check_nested_list_struct(cls, pandera_model: pla.DataFrameModel, data: PolarsData):
        """Use this in a custom check. Pass the nested model as pandera_model."""
        column_name = data.key

        try:
            # fmt: off
            unnested_column = (
                data.lazyframe
                .select(column_name)
                .filter(pl.col(column_name).list.len() > 0) # see: https://github.com/pola-rs/polars/issues/14381
                .explode(column_name)
                .unnest(column_name)
            )
            # fmt: on
        except pl.exceptions.PanicException:
            return data.lazyframe.select(pl.lit(False))
        except Exception:
            return data.lazyframe.select(pl.lit(False))

        try:
            nested_cls = cls.get_nested_range(column_name)
            nested_cls.validate(unnested_column, lazy=True)
        except pl.exceptions.PanicException:
            return data.lazyframe.select(pl.lit(False))
        except Exception:
            return data.lazyframe.select(pl.lit(False))

        return data.lazyframe.select(pl.lit(True))

    @classmethod
    def _check_nested_struct(cls, pandera_model: pla.DataFrameModel, data: PolarsData):
        """Use this in a custom check. Pass the nested model as pandera_model."""
        column_name = data.key

        try:
            # fmt: off
            unnested_column = (
                data.lazyframe
                .select(column_name)
                .unnest(column_name)
            )
            # fmt: on
        except pl.exceptions.PanicException:
            return data.lazyframe.select(pl.lit(False))

        try:
            nested_cls = cls.get_nested_range(column_name)
            nested_cls.validate(unnested_column, lazy=True)
        except pl.exceptions.PanicException:
            return data.lazyframe.select(pl.lit(False))
        except Exception:
            return data.lazyframe.select(pl.lit(False))

        return data.lazyframe.select(pl.lit(True))

    @classmethod
    def simple_dict_to_list_of_structs_expr(cls, column_name: str, id_col, other_col, polars_schema) -> pl.Expr:
        if len(polars_schema.columns.keys()) == 2:
            id_col_type = polars_schema.columns[id_col]
            other_col_type = polars_schema.columns[other_col]

            polars_struct = pl.Struct({id_col: id_col_type.dtype.type, other_col: other_col_type.dtype.type})
        else:
            struct_items = {}
            for k, v in polars_schema.columns.items():
                if v.dtype.type == pl.Object:
                    v.dtype.type = pl.Struct
                else:
                    struct_items[k] = v.dtype.type
            polars_struct = pl.Struct(struct_items)

        def map_lamb(e):
            arr = []

            for x, y in e.items():

                if isinstance(y, dict) and (set(y.keys()) <= set(polars_schema.columns.keys())):
                    d = y
                    for c in polars_schema.columns.keys():
                        if c not in d:
                            d[c] = None
                else:
                    d = {id_col: x, other_col: y}
                arr.append(d)
            return arr

        # fmt: off
        return (
            pl.col(column_name)
            .map_elements(
                map_lamb,
                skip_nulls=True,
                return_dtype=pl.List(polars_struct)
            )
        )
        # fmt: on

    @classmethod
    def transform_to_cannonical(cls, df: pl.DataFrame, object_to_validate) -> pl.DataFrame:
        select_expr = []

        for column_name, column in cls.to_schema().columns.items():
            dtype = column.properties["dtype"]
            required = column.properties["required"]

            if required or column_name in object_to_validate:
                if dtype.type == pl.Struct:
                    if cls._INLINE_FORM[column_name] == "simple_dict":
                        df = df.with_columns(cls.simple_dict_to_list_of_structs_expr(column_name, "XIDX", "XOTHERX"))
                    else:
                        # nested_cls_name = cls._NESTED_RANGES[column_name]
                        # nested_cls = getattr(inspect.getmodule(cls), nested_cls_name)
                        nested_cls = cls.get_nested_range(column_name)
                        select_expr.append(
                            pl.col(column_name).cast(nested_cls.generate_polars_schema(object_to_validate[column_name]))
                        )
                else:
                    select_expr.append(pl.col(column_name).cast(dtype.type))

        return df

    @classmethod
    def get_nested_range(cls, column_name):
        """Resolve a nested class range at runtime.

        Nested classes are not stored in the pandera schema,
        but rather in the _NESTED_RANGES dictionary as strings.
        """
        nested_cls_name = cls._NESTED_RANGES[column_name]
        shared_model_module = inspect.getmodule(cls)
        nested_cls = getattr(shared_model_module, nested_cls_name)

        return nested_cls

    @classmethod
    def generate_polars_schema(cls, object_to_validate, parser=False) -> dict:
        """Creates a nested PolaRS schema suitable for loading the object_to_validate.
        Optional columns that are not present in the data are omitted.
        This approach is only suitable to enable the test fixtures.
        """
        polars_schema = {}

        for column_name, column in cls.to_schema().columns.items():
            dtype = column.properties["dtype"]
            required = column.properties["required"]

            if required or column_name in object_to_validate:
                if dtype.type in [pl.Struct, pl.List]:
                    inline_form = cls._INLINE_FORM[column_name]
                    if inline_form == "simple_dict":
                        polars_schema[column_name] = pl.Object  # make this a struct and make the nested non-
                    else:
                        nested_cls = cls.get_nested_range(column_name)
                        if inline_form == "inlined_dict":
                            if parser:
                                nested_schema = nested_cls.generate_polars_schema(object_to_validate[column_name])
                                polars_schema[column_name] = pl.Struct(nested_schema)
                            else:
                                polars_schema[column_name] = pl.Struct
                        elif inline_form == "inlined_list_dict":
                            if parser:
                                # MAYBE NEED TO ACTUALLY CHECK THE NESTED TYPE HERE
                                nested_schema = nested_cls.generate_polars_schema(object_to_validate[column_name])
                                polars_schema[column_name] = pl.Struct(nested_schema)
                                # transformed form
                                polars_schema[column_name] = pl.Struct
                else:
                    polars_schema[column_name] = dtype.type

        return polars_schema
