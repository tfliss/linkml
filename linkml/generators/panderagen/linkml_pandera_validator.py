import inspect
from functools import wraps

import narwhals as nw
import pandera.polars as pla
import polars as pl
from pandera.api.polars.types import PolarsData
import pandera
from linkml.generators.panderagen.transforms.simple_dict_model_transform import SimpleDictModelTransform
from linkml.generators.panderagen.transforms.collection_dict_model_transform import CollectionDictModelTransform
from linkml.generators.panderagen.transforms.list_dict_model_transform import ListDictModelTransform


def handle_validation_exceptions(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except pl.exceptions.PanicException:
            data = args[2] if len(args) > 2 else kwargs.get('data')
            return data.lazyframe.select(pl.lit(False))
        except pandera.errors.SchemaError as e:
            raise e
        except Exception:
            data = args[2] if len(args) > 2 else kwargs.get('data')
            return data.lazyframe.select(pl.lit(False))
    return wrapper


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
    def _prepare_simple_dict(cls, pandera_model: pla.DataFrameModel, data : PolarsData):
        """Returns just the simple dict column tranformed to an inlined list form

           note that this method uses collect and iter_rows so is very inefficient
        """
        column_name = data.key
        polars_schema = cls.get_nested_range(column_name).to_schema()
        (id_column, other_column) = cls._simple_dict_fields(column_name, pandera_model)

        simple_dict_transformer = SimpleDictModelTransform(polars_schema, id_column, other_column)

        one_column_df = data.lazyframe.select(pl.col(column_name)).collect()

        list_of_structs = [
            simple_dict_transformer.transform(e)
            for [e] in one_column_df.iter_rows()
        ]

        return pl.DataFrame(
            pl.Series(list_of_structs).alias(column_name)
        )
    
    @classmethod
    @handle_validation_exceptions
    def _check_simple_dict(cls, pandera_model: pla.DataFrameModel, data: PolarsData):
        """
           The 'simple dict' format, in which the key serves as a local identifier is not a good match for a PolaRS
           DataFrame. At present the format is 
        """
        df = cls._prepare_simple_dict(pandera_model, data)

        column_name = data.key

        df = (
            df.lazy()
            .explode(column_name)
            .unnest(column_name)
            .collect()
        )

        nested_cls = cls.get_nested_range(column_name)
        nested_cls.validate(df)
        return data.lazyframe.select(pl.lit(True))

    @classmethod
    @handle_validation_exceptions
    def _check_collection_struct(cls, pandera_model: pla.DataFrameModel, data: PolarsData):
        column_name = data.key
        nested_cls = cls.get_nested_range(column_name)
        
        df = CollectionDictModelTransform.prepare_dataframe(data, column_name, nested_cls)

        df = (
            df.lazy()
            .filter(pl.col(column_name).list.len() > 0)
            .explode(column_name)
            .unnest(column_name)
            .collect()
        )

        nested_cls.validate(df)
        return data.lazyframe.select(pl.lit(True))


    @classmethod
    @handle_validation_exceptions
    def _check_nested_list_struct(cls, pandera_model: pla.DataFrameModel, data: PolarsData):
        """Use this in a custom check. Pass the nested model as pandera_model."""
        column_name = data.key
        nested_cls = cls.get_nested_range(column_name)
        
        df = ListDictModelTransform.prepare_dataframe(data, column_name, nested_cls)

        try:
            df = (
                df.lazy()
                .filter(pl.col(column_name).list.len() > 0)
                .explode(column_name)
                .unnest(column_name)
                .collect()
            )
        except (pl.exceptions.PanicException, Exception):
            df = cls._unnest_struct(column_name, data.lazyframe).collect()

        nested_cls.validate(df)
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
    @handle_validation_exceptions
    def _check_nested_struct(cls, pandera_model: pla.DataFrameModel, data: PolarsData):
        """Use this in a custom check. Pass the nested model as pandera_model."""
        column_name = data.key

        unnested_column = cls._unnest_struct(column_name, data.lazyframe).collect().lazy()
        nested_cls = cls.get_nested_range(column_name)
        nested_cls.validate(unnested_column, lazy=True)
        return data.lazyframe.select(pl.lit(True))

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
