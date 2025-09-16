import inspect
import logging
from functools import wraps

import pandera
import polars as pl
from pandera.api.polars.types import PolarsData

from linkml.generators.panderagen.transforms import (
    CollectionDictModelTransform,
    ListDictModelTransform,
    NestedStructModelTransform,
    SimpleDictModelTransform,
)

logger = logging.getLogger(__name__)


def handle_validation_exceptions(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except pl.exceptions.PanicException:
            data = args[2] if len(args) > 2 else kwargs.get("data")
            return data.lazyframe.select(pl.lit(False))
        except pandera.errors.SchemaError as e:
            raise e
        except Exception:
            data = args[2] if len(args) > 2 else kwargs.get("data")
            return data.lazyframe.select(pl.lit(False))

    return wrapper


class LinkmlPanderaValidator:
    @classmethod
    def get_id_column_name(cls):
        return cls._id_name

    @classmethod
    def _simple_dict_fields(cls, column_name):
        details = cls._INLINE_DETAILS[column_name]  # <-- THESE ARE GOING ON THE OUTER CLASS

        return (details["id"], details["other"])

    @classmethod
    def _prepare_simple_dict(cls, data: PolarsData, polars_schema: pl.Schema):
        """Returns just the simple dict column transformed to an inlined list form

        note that this method uses collect and iter_rows so is very inefficient
        """
        column_name = data.key

        (id_column, other_column) = cls._simple_dict_fields(column_name)

        simple_dict_transformer = SimpleDictModelTransform(polars_schema, id_column, other_column)

        one_column_df = data.lazyframe.select(pl.col(column_name)).collect()

        list_of_structs = [simple_dict_transformer.transform(e) for [e] in one_column_df.iter_rows()]

        return pl.DataFrame(pl.Series(list_of_structs).alias(column_name))

    @classmethod
    @handle_validation_exceptions
    def _check_simple_dict(cls, data: PolarsData, polars_schema: pl.Schema):
        """
        The 'simple dict' format, in which the key serves as a local identifier is not a good match for a PolaRS
        DataFrame. At present the format is
        """
        column_name = data.key

        df = cls._prepare_simple_dict(data, polars_schema)

        simple_transform = SimpleDictModelTransform(polars_schema, *cls._simple_dict_fields(column_name))
        df = simple_transform.explode_unnest_dataframe(df, column_name)

        nested_cls = cls.get_nested_range(column_name)
        nested_cls.validate(df)
        return data.lazyframe.select(pl.lit(True))

    @classmethod
    @handle_validation_exceptions
    def _check_collection_struct(cls, data: PolarsData, nested_cls: type, polars_schema: pl.Schema):
        column_name = data.key

        df = CollectionDictModelTransform.prepare_dataframe(data, column_name, nested_cls)

        collection_transform = CollectionDictModelTransform(polars_schema, nested_cls.get_id_column_name())
        df = collection_transform.explode_unnest_dataframe(df, column_name)

        nested_cls.validate(df)
        return data.lazyframe.select(pl.lit(True))

    @classmethod
    @handle_validation_exceptions
    def _check_nested_list_struct(cls, data: PolarsData, nested_cls: type, polars_schema: pl.Schema):
        """Use this in a custom check. Pass the nested model as pandera_model."""
        column_name = data.key

        df = ListDictModelTransform.prepare_dataframe(data, column_name, nested_cls)

        list_transform = ListDictModelTransform(polars_schema)
        df = list_transform.explode_unnest_dataframe(df, column_name, data)

        nested_cls.validate(df)
        return data.lazyframe.select(pl.lit(True))

    @classmethod
    @handle_validation_exceptions
    def _check_nested_struct(cls, data: PolarsData, nested_cls: type, polars_schema: pl.Schema):
        """Use this in a custom check. Pass the nested model as pandera_model."""
        try:
            column_name = data.key

            df = NestedStructModelTransform.prepare_dataframe(data, column_name, nested_cls)
            nested_transform = NestedStructModelTransform(polars_schema)  # nested_cls.to_schema())
            df = nested_transform.explode_unnest_dataframe(df, column_name)

            nested_cls.validate(df)
        except Exception as e:
            logger.info(f"Error validating {data.key}")
            raise e

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
