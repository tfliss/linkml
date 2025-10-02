import logging

import polars as pl
from pandera.api.polars.types import PolarsData
from pandera.errors import SchemaError

from linkml.generators.panderagen.transforms import (
    CollectionDictModelTransform,
    ListDictModelTransform,
    NestedStructModelTransform,
    SimpleDictModelTransform,
)

logger = logging.getLogger(__name__)


class LinkmlPanderaValidator:
    @classmethod
    def get_id_column_name(cls):
        """
        _id_name is present in the implementing class
        """
        return cls._id_name

    @classmethod
    def _simple_dict_to_list_of_structs(cls, one_column_df, simple_dict_transformer):
        """
        TODO: combine these with the new_transforms version
        """
        for [e] in one_column_df.iter_rows():
            transformed = list(simple_dict_transformer.transform(e))  # generator
            if len(transformed) > 0:
                yield transformed

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

        list_of_structs = list(cls._simple_dict_to_list_of_structs(one_column_df, simple_dict_transformer))
        return pl.DataFrame(pl.Series(list_of_structs, dtype=polars_schema_dict, strict=True).alias(column_name))

    @classmethod
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
    def _check_collection_struct(
        cls, data: PolarsData, nested_cls: type, polars_schema: pl.Schema, polars_schema_struct
    ):
        column_name = data.key

        collection_transform = CollectionDictModelTransform(nested_cls, polars_schema, polars_schema_struct)
        df = collection_transform.prepare_dataframe(data, column_name)
        if df.schema[column_name] != pl.List(pl.Struct(polars_schema)):
            raise SchemaError(
                polars_schema, df, f"Schema mismatch for {column_name}: {df.schema[column_name]} != {polars_schema}"
            )
        df = collection_transform.explode_unnest_dataframe(df, column_name)

        nested_cls.validate(df)
        return data.lazyframe.select(pl.lit(True))

    @classmethod
    def _check_nested_list_struct(cls, data: PolarsData, nested_cls: type, polars_schema):
        """Use this in a custom check. Pass the nested model as pandera_model."""
        column_name = data.key

        list_transform = ListDictModelTransform(polars_schema)

        df = list_transform.prepare_dataframe(data, column_name, nested_cls)

        # TODO: form of polars_schema needs to be more regular wrt container
        if df.schema[column_name] != pl.List(pl.Struct(polars_schema)):
            raise SchemaError(
                polars_schema, df, f"Schema mismatch for {column_name}: {df.schema[column_name]} != {polars_schema}"
            )

        df = list_transform.explode_unnest_dataframe(df, column_name, data)
        nested_cls.validate(df)

        return data.lazyframe.select(pl.lit(True))

    @classmethod
    def _check_nested_struct(cls, data: PolarsData, nested_cls: type, polars_schema: pl.Schema):
        """Use this in a custom check. Pass the nested model as pandera_model."""
        column_name = data.key

        df = NestedStructModelTransform.prepare_dataframe(data, column_name, nested_cls)
        nested_transform = NestedStructModelTransform(pl.Schema(polars_schema))  # nested_cls.to_schema())

        if df.schema[column_name] != pl.Struct(polars_schema):
            raise SchemaError(
                polars_schema, df, f"Schema mismatch for {column_name}: {df.schema[column_name]} != {polars_schema}"
            )

        df = nested_transform.explode_unnest_dataframe(df, column_name)
        nested_cls.validate(df)

        return data.lazyframe.select(pl.lit(True))
