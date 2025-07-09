import inspect

import narwhals as nw
import pandera.polars as pla
import polars as pl
from pandera.api.polars.types import PolarsData
import pandera


class SimpleDictModelTransform:
    """This class assists in converting a LinkML 'simple dict' inline column
       into a form that is better for representing in a PolaRS dataframe and
       validating with a Pandera model.
    """

    def __init__(self, polars_schema, id_col, other_col):
        self.polars_schema = polars_schema
        """A polars schema representing a simple dict column"""        

        self.id_col = id_col
        """The ID column in the sense of a LinkML inline simple dict"""
        
        self.other_col = other_col
        """The 'other' column in the sense of a LinkML inline simple dict"""
        
        self.id_col_type = None
        self.other_col_type = None
        self.polars_struct = self._build_polars_struct()
        """A pl.Struct representing the schema of the other range."""

    def _build_polars_struct_simple(self):
        """Handles the two column (id, other) form of the simple dict
        """
        self.id_col_type = self.polars_schema.columns[self.id_col].dtype.type
        self.other_col_type = self.polars_schema.columns[self.other_col].dtype.type

        return pl.Struct({self.id_col: self.id_col_type, self.other_col: self.other_col_type})

    def _build_polars_struct_complex(self):
        """Handles the non-two-column simple dict cases.
        """
        struct_items = {}
        for k, v in self.polars_schema.columns.items():
            if v.dtype.type == pl.Object:
                v.dtype.type = pl.Struct
            else:
                struct_items[k] = v.dtype.type
        return pl.Struct(struct_items)

    def _build_polars_struct(self):
        if len(self.polars_schema.columns.keys()) == 2:
            return self._build_polars_struct_simple()
        else:
            return self._build_polars_struct_complex()

    def simple_dict_to_list_of_structs(self, linkml_simple_dict):
        """ { 'A': 1, 'B': 2, ... } -> [{'id': 'other': 1}, {'id': 'B', 'other': 2}, ...]
        
           An inefficient conversion (relative to native PolaRS operations)
           from a simple dict form to a dataframe struct column.

           e : dict
               e is a single row entry in a dataframe column (one cell), which itself is a dict.
               The value entries of e may also be dicts.
        """
        arr = []
        for id_value, range_value in linkml_simple_dict.items():
            if isinstance(range_value, dict) and (set(range_value.keys()) <= set(self.polars_schema.columns.keys())):
                range_dict = range_value
                range_dict[self.id_col] = id_value
                for column_key in self.polars_schema.columns.keys():
                    if column_key not in range_dict:
                        range_dict[column_key] = None
            else:
                range_dict = {self.id_col: id_value, self.other_col: range_value}
            arr.append(range_dict)

        return arr

    def list_dtype(self):
        return pl.List(self.polars_struct)


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

        transformer = SimpleDictModelTransform(polars_schema, id_column, other_column)

        one_column_df = data.lazyframe.select(pl.col(column_name)).collect()

        list_of_structs = [
            transformer.simple_dict_to_list_of_structs(e)
            for [e] in one_column_df.iter_rows()
        ]

        return pl.DataFrame(
            pl.Series(list_of_structs).alias(column_name)
        )
    
    @classmethod
    def _check_simple_dict(cls, pandera_model: pla.DataFrameModel, data: PolarsData):
        """
           The 'simple dict' format, in which the key serves as a local identifier is not a good match for a PolaRS
           DataFrame. At present the format is 
        """
        df = cls._prepare_simple_dict(pandera_model, data)

        column_name = data.key

        df = (
            df.lazy()
            #.select(pl.col(column_name))
            .explode(column_name)
            .unnest(column_name)
            .collect()
        )

        try:
            nested_cls = cls.get_nested_range(column_name)
            nested_cls.validate(df)
        except pandera.errors.SchemaError as e:
            raise e
        except pandera.errors.SchemaError as e:
            raise e
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
