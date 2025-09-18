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
        except pandera.errors.SchemaError as e:
            raise e
        except Exception as e:
            logger.info(e)
            data = args[2] if len(args) > 2 else kwargs.get("data")
            return data.lazyframe.select(pl.lit(False))

    return wrapper


class LinkmlPanderaValidator:
    @classmethod
    def get_id_column_name(cls):
        return cls._id_name

    @classmethod
    def _prepare_simple_dict(
        cls, data: PolarsData, id_col: str, other_col: str, polars_schema: pl.Schema, polars_schema_dict
    ):
        """Returns just the simple dict column transformed to an inlined list form

        note that this method uses collect and iter_rows so is very inefficient
        """
        column_name = data.key

        simple_dict_transformer = SimpleDictModelTransform(id_col, other_col, polars_schema, polars_schema_dict)

        # TODO: check if need to do the filter for null here
        one_column_df = data.lazyframe.select(pl.col(column_name)).collect()

        # TODO: try doing this as a generator
        list_of_structs = []
        for [e] in one_column_df.iter_rows():
            transformed = simple_dict_transformer.transform(e)
            if len(transformed) > 0:
                list_of_structs.append(transformed)

        if len(list_of_structs) == 0:
            list_of_structs = None

        return pl.DataFrame(pl.Series(list_of_structs, dtype=polars_schema_dict, strict=True).alias(column_name))

    @classmethod
    @handle_validation_exceptions
    def _check_simple_dict(
        cls,
        data: PolarsData,
        nested_cls: type,
        id_col: str,
        other_col: str,
        polars_schema: pl.Schema,
        polars_schema_dict,
    ):
        """
        The 'simple dict' format, in which the key serves as a local identifier is not a good match for a PolaRS
        DataFrame. At present the format is
        """
        column_name = data.key

        df = cls._prepare_simple_dict(data, id_col, other_col, polars_schema, polars_schema_dict)
        simple_transform = SimpleDictModelTransform(id_col, other_col, polars_schema, polars_schema_dict)
        df = simple_transform.explode_unnest_dataframe(df, column_name)

        nested_cls.validate(df)
        return data.lazyframe.select(pl.lit(True))

    @classmethod
    @handle_validation_exceptions
    def _check_collection_struct(
        cls, data: PolarsData, nested_cls: type, polars_schema: pl.Schema, polars_schema_struct
    ):
        column_name = data.key

        collection_transform = CollectionDictModelTransform(nested_cls, polars_schema, polars_schema_struct)
        df = collection_transform.prepare_dataframe(data, column_name)
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
