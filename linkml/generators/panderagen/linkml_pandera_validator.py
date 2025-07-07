import inspect

import narwhals as nw
import pandera.polars as pla
import polars as pl
from pandera.api.polars.types import PolarsData


class LinkmlPanderaValidator:

    @classmethod
    def get_id_column_name(cls):
        return cls._id_name

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
    def _unnest_list_struct(cls, column_name: str, df):
        """Use this in a custom check. Pass the nested model as pandera_model."""

        # fmt: off
        unnested_column = (
            df
            .select(column_name)
            .filter(pl.col(column_name).list.len() > 0) # see: https://github.com/pola-rs/polars/issues/14381
            .explode(column_name)
            .unnest(column_name)
        )
        # fmt: on

        return unnested_column

    @classmethod
    def collection_struct_mapper(cls, column_name: str):
        """used in a collection struct to convert the dict object to a list of structs via map_elements.
           the name of the id column needs to be looked up in the schema.
        """
        nested_cls = cls.get_nested_range(column_name)
        id_column = nested_cls.get_id_column_name()

        def mapping_lambda(x):
            arr = []

            for k,v in x.items():
                if k not in v:
                    v[id_column] = k
                arr.append(v)

            return arr

        return mapping_lambda

    @classmethod
    def _check_collection_struct(cls, pandera_model: pla.DataFrameModel, data: PolarsData):
        df = data.lazyframe
        column_name = data.key

        schema = pandera_model.generate_polars_schema_simple()

        # fmt: off
        unnested_column =  (
            df
            .select(
                pl.col(column_name)
                .map_elements(
                    cls.collection_struct_mapper(column_name),
                    skip_nulls=True,
                    return_dtype=pl.List(schema)
                )
            )
            .filter(pl.col(column_name).list.len() > 0) # see: https://github.com/pola-rs/polars/issues/14381
            .explode(column_name)
            .unnest(column_name)
        )
        # fmt: on

        try:
            nested_cls = cls.get_nested_range(column_name)
            nested_cls.validate(unnested_column, lazy=True)
        except (pl.exceptions.PanicException, Exception):
            return data.lazyframe.select(pl.lit(False))


        return data.lazyframe.select(pl.lit(True))


    @classmethod
    def _check_nested_list_struct(cls, pandera_model: pla.DataFrameModel, data: PolarsData):
        """Use this in a custom check. Pass the nested model as pandera_model."""
        column_name = data.key

        try:
            unnested_column = cls._unnest_list_struct(column_name, data.lazyframe).collect().lazy()
        except (pl.exceptions.PanicException, Exception):
            try:
                unnested_column = cls._unnest_struct(column_name, data.lazyframe).collect().lazy()
            except (pl.exceptions.PanicException, Exception):
                return data.lazyframe.select(pl.lit(False))

        try:
            nested_cls = cls.get_nested_range(column_name)
            nested_cls.validate(unnested_column, lazy=True)
        except (pl.exceptions.PanicException, Exception):
            return data.lazyframe.select(pl.lit(False))

        return data.lazyframe.select(pl.lit(True))


    @classmethod
    def _unnest_struct(cls, column_name: str, df):
        """Use this in a custom check. Pass the nested model as pandera_model."""

        # fmt: off
        unnested_column = (
            df
            .select(column_name)
            .unnest(column_name)
        )
        # fmt: on

        return unnested_column

    @classmethod
    def _check_nested_struct(cls, pandera_model: pla.DataFrameModel, data: PolarsData):
        """Use this in a custom check. Pass the nested model as pandera_model."""
        column_name = data.key

        try:
            unnested_column = cls._unnest_struct(column_name, data.lazyframe).collect().lazy()
        except pl.exceptions.PanicException:
            return data.lazyframe.select(pl.lit(False))

        try:
            nested_cls = cls.get_nested_range(column_name)
            nested_cls.validate(unnested_column, lazy=True)
        except (pl.exceptions.PanicException, Exception):
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
                    d[id_col] = x
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
    def generate_polars_schema_simple(cls):
        # This is not nesting or list aware, so needs to be aligned with the other method
        return pl.Struct(
            {
                k: v.dtype.type for k, v in cls.to_schema().columns.items()
            }
        )

    @classmethod
    def generate_polars_schema(cls, object_to_validate, parser=False) -> dict:
        """Creates a nested PolaRS schema suitable for loading the object_to_validate.
        Optional columns that are not present in the data are omitted.
        This approach is only suitable to enable the test fixtures.
        """
        polars_schema = {}

        if isinstance(object_to_validate, list):
            object_to_validate = object_to_validate[0]

        for column_name, column in cls.to_schema().columns.items():
            dtype = column.properties["dtype"]
            required = column.properties["required"]

            if required or column_name in object_to_validate:
                if dtype.type in [pl.Struct, pl.List]:  # maybe use inline form directly here
                    inline_form = cls._INLINE_FORM.get(column_name, "not_inline")
                    if inline_form == "simple_dict":
                        polars_schema[column_name] = pl.Object  # make this a struct and make the nested non-
                    elif inline_form == "not_inline":
                        polars_schema[column_name] = dtype.type
                    else:
                        nested_cls = cls.get_nested_range(column_name)
                        if inline_form == "inlined_dict":
                            if parser:
                                nested_schema = nested_cls.generate_polars_schema(
                                    object_to_validate[column_name],
                                    parser
                                )
                                polars_schema[column_name] = pl.Struct(nested_schema)
                            else:
                                polars_schema[column_name] = pl.Struct
                        elif inline_form == "inlined_list_dict":
                            if parser:
                                nested_schema = nested_cls.generate_polars_schema(
                                    object_to_validate[column_name],
                                    parser
                                )
                                polars_schema[column_name] = pl.List(pl.Struct(nested_schema))
                            else:
                                # transformed form
                                polars_schema[column_name] = pl.List
                else:
                    polars_schema[column_name] = dtype.type

        return polars_schema
