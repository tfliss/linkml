import logging
import sys

import pytest

from linkml.generators.panderagen import PanderaDataframeGenerator
from linkml.generators.panderagen.polars_schema.polars_schema_dataframe_generator import PolarsSchemaDataframeGenerator

logger = logging.getLogger(__file__)

pl = pytest.importorskip("polars", minversion="1.0", reason="Polars >= 1.0 not installed")
np = pytest.importorskip("numpy", reason="NumPY not installed")


@pytest.fixture(scope="module")
def N():
    """Number of rows in the test dataframes, 1M is enough to be real but not strain most machines."""
    return 1000


@pytest.fixture(scope="module")
def synthetic_flat_dataframe_model():
    return """\
id: https://w3id.org/linkml/examples/pandera_constraints
name: test_pandera_constraints
prefixes:
  linkml: https://w3id.org/linkml/
  ex: https://w3id.org/linkml/examples/pandera_constraints/
imports:
  - linkml:types
default_range: string
default_prefix: ex

classes:

  AnyType:
    description: the magic class_uri makes this map to linkml Any or polars Object
    class_uri: linkml:Any

  ColumnType:
    description: Nested in a column
    attributes:
      id:
        identifier: true
        range: string
      x:
        range: integer
        required: true
      y:
        range: integer
        required: true

  SimpleDictType:
    description: Nested as a simple dict
    attributes:
      id:
        identifier: True
        range: string
      x:
        range: integer
        required: true

  PanderaSyntheticTable:
    description: A flat table with a reasonably complete assortment of datatypes.
    attributes:
      identifier_column:
        description: identifier
        identifier: true
        range: integer
        required: true
      bool_column:
        description: test boolean column
        range: boolean
        required: true
        #ifabsent: true
      integer_column:
        description: test integer column with min/max values
        range: integer
        required: true
        minimum_value: 0
        maximum_value: 999
        #ifabsent: int(5)
      float_column:
        description: test float column
        range: float
        required: true
        #ifabsent: float(2.3)
      string_column:
        description: test string column
        range: string
        required: true
        pattern: "^(this)|(that)|(whatever)$"
        #ifabsent: string("whatever")
      date_column:
        description: test date column
        range: date
        required: true
        #ifabsent: date("2020-01-31")
      datetime_column:
        description: test datetime column
        range: datetime
        required: true
        #ifabsent: datetime("2020-01-31 03:23:57")
      enum_column:
        description: test enum column
        range: SyntheticEnum
        required: true
      ontology_enum_column:
        description: test enum column with ontology values
        range: SyntheticEnumOnt
        required: true
        #ifabsent: SyntheticEnumOnt(ANIMAL)
      multivalued_column:
        description: one-to-many form
        range: integer
        required: true
        multivalued: true
        inlined_as_list: true
      # multivalued_one_many_column:
      #   description: list form
      #   range: integer
      #   required: true
      #   multivalued: true
      any_type_column:
        description: needs to have type object
        range: AnyType
        required: true
      cardinality_column:
        description: check cardinality
        range: integer
        required: true
        minimum_cardinality: 1
        maximum_cardinality: 1
      inlined_as_object_column:
        description: test column that is a directly nested single object (no dictionary collection)
        range: ColumnType
        required: true
        inlined: true
        multivalued: false
      foreign_key_object_column:
        description: test column that is an association to another table
        range: ColumnType
        required: true
        inlined: false
        multivalued: false
      inlined_class_column:
        description: test column with another class inlined as a struct
        range: ColumnType
        required: true
        inlined: true
        inlined_as_list: false
        multivalued: true
      inlined_as_list_column:
        description: test column with another class inlined as a list
        range: ColumnType
        required: true
        inlined: true
        inlined_as_list: true
        multivalued: true
      inlined_simple_dict_column:
        description: test column inlined using simple dict form
        range: SimpleDictType
        multivalued: true
        inlined: true
        inlined_as_list: false
        required: true


enums:
  SyntheticEnum:
    description: simple enum for tests
    permissible_values:
      ANIMAL:
      VEGETABLE:
      MINERAL:

  SyntheticEnumOnt:
    description: ontology enum for tests
    permissible_values:
      fiction: ex:000001
      non fiction: ex:000002
"""


@pytest.fixture(scope="module")
def synthetic_schema(synthetic_flat_dataframe_model):
    generator = PolarsSchemaDataframeGenerator(synthetic_flat_dataframe_model)
    generator.template_file = "polars_schema.jinja2"
    generator.template_path = "panderagen_polars_schema"

    return generator


@pytest.fixture(scope="module")
def compiled_synthetic_schema_module(synthetic_schema):
    logger.info(f"{synthetic_schema.serialize()}")

    return synthetic_schema.compile_dataframe_model("panderagen_polars_schema")


@pytest.fixture(scope="module")
def synthetic_pandera_schema(synthetic_flat_dataframe_model):
    return PanderaDataframeGenerator(synthetic_flat_dataframe_model)


@pytest.fixture(scope="module")
def compiled_synthetic_pandera_schema_module(compiled_synthetic_schema_module, synthetic_pandera_schema):
    del compiled_synthetic_schema_module  # suppress warning

    logger.info(f"{synthetic_pandera_schema.serialize()}")
    yield synthetic_pandera_schema.compile_dataframe_model("panderagen_class_based")

    # unload the modules used in this testing
    sys.modules.pop("panderagen_polars_schema", None)
    sys.modules.pop("panderagen_class_based", None)


@pytest.fixture(scope="module")
def column_type_instances():
    """valid ColumnType instances that can be used in tests"""
    return [
        {
            "id": "thing_one",
            "x": 1111,
            "y": 2222,
        },
        {
            "id": "thing_two",
            "x": 3333,
            "y": 4444,
        },
    ]


@pytest.fixture(scope="module")
def invalid_column_type_instances():
    """invalid (float values) ColumnType instances that can trigger failures."""
    return [
        {
            "id": "thing_one",
            "x": 1111.1,
            "y": 2222.2,
        },
        {
            "id": "thing_two",
            "x": 3333.3,
            "y": 4444.4,
        },
    ]


@pytest.fixture(scope="module")
def valid_inlined_dict_column_expression(column_type_instances):
    """synthetic data that conforms to the inlined_class_column schema
    using polars expression API.
    """
    # fmt: off
    return {
        "thing_one": column_type_instances[0],
        "thing_two": column_type_instances[1]
    }
    # fmt: on


@pytest.fixture(scope="module")
def invalid_inlined_dict_column_expression(invalid_column_type_instances):
    """synthetic data that conforms to the inlined_class_column schema
    using polars expression API.
    """
    # fmt: off
    return {
        "thing_one": invalid_column_type_instances[0],
        "thing_two": invalid_column_type_instances[1]
    }
    # fmt: on


@pytest.fixture(scope="module")
def valid_simple_dict_column_expression():
    """synthetic data that conforms to the inlined_simple_dict_column schema."""
    return {"A": 1, "B": 2, "C": 3}


@pytest.fixture(scope="module")
def invalid_simple_dict_column_expression():
    """synthetic data with float values that does not conform to the inlined_simple_dict_column schema."""
    return {"A": 1.1, "B": 2.2, "C": 3.3}


@pytest.fixture(scope="module")
def big_synthetic_dataframe(
    N,
    column_type_instances,
    valid_inlined_dict_column_expression,
    valid_simple_dict_column_expression,
    compiled_synthetic_schema_module,
):
    """Construct a reasonably sized dataframe that complies with the PanderaSyntheticTable model"""
    test_enum = pl.Enum(["ANIMAL", "VEGETABLE", "MINERAL"])
    test_ont_enum = pl.Enum(["fiction", "non fiction"])

    # fmt: off
    df = (
        pl.DataFrame(
            {
                "identifier_column": pl.Series(np.arange(0, N), dtype=pl.Int64),
                "bool_column": pl.Series(np.random.choice([True, False], size=N), dtype=pl.Boolean),
                "integer_column": pl.Series(np.random.choice(range(100), size=N), dtype=pl.Int64),
                "float_column": pl.Series(np.random.choice([1.0, 2.0, 3.0], size=N), dtype=pl.Float64),
                "string_column": np.random.choice(["this", "that"], size=N),
                "date_column": pl.Series(
                    np.random.choice(["2021-03-27", "2021-03-28"], size=N),
                    dtype=pl.Date,
                    strict=False
                ),
                "datetime_column": pl.Series(
                    np.random.choice(["2021-03-27T03:00:00", "2021-03-28T03:00:00"], size=N),
                    dtype=pl.Datetime(time_unit='us', time_zone=None),
                    strict=False
                ),
                  "enum_column": pl.Series(
                      np.random.choice(["ANIMAL", "VEGETABLE", "MINERAL"], size=N),
                      dtype=test_enum,
                      strict=False
                  ),
                  "ontology_enum_column": pl.Series(
                      np.random.choice(["fiction", "non fiction"], size=N),
                      dtype=test_ont_enum,
                      strict=False
                  ),
                "multivalued_column": [[1, 2, 3]] * N,
                "any_type_column": [1] * N,
                "cardinality_column": np.arange(1, N+1),
                "inlined_as_object_column": [ column_type_instances[0] ] * N,
                "foreign_key_object_column": [ "thing_one" ] * N,
                "inlined_simple_dict_column": [valid_simple_dict_column_expression] * N,
                "inlined_as_list_column": [ column_type_instances ] * N,
                "inlined_class_column": [ valid_inlined_dict_column_expression ] * N, # is multivalued collection dict
            },
            schema=compiled_synthetic_schema_module.PanderaSyntheticTable
        )
    )
    # fmt: on

    logger.info(df)

    return df
