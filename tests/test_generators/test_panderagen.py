import logging
import re

import pytest

from linkml.generators.panderagen import PanderaGenerator

logger = logging.getLogger(__name__)


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
        identifier: True
        range: string
      x:
        range: integer
      y:
        range: integer

  PanderaSyntheticTable:
    description: A flat table with a reasonably complete assortment of datatypes.
    attributes:
      identifier_column:
        description: identifier
        identifier: True
        range: integer
        required: True
      bool_column:
        description: test boolean column
        range: boolean
        required: True
        #ifabsent: True
      integer_column:
        description: test integer column with min/max values
        range: integer
        required: True
        minimum_value: 0
        maximum_value: 999
        #ifabsent: int(5)
      float_column:
        description: test float column
        range: float
        required: True
        #ifabsent: float(2.3)
      string_column:
        description: test string column
        range: string
        required: True
        pattern: "^(this)|(that)|(whatever)$"
        #ifabsent: string("whatever")
      date_column:
        description: test date column
        range: date
        required: True
        #ifabsent: date("2020-01-31")
      datetime_column:
        description: test datetime column
        range: datetime
        required: True
        #ifabsent: datetime("2020-01-31 03:23:57")
      enum_column:
        description: test enum column
        range: SyntheticEnum
        required: True
      ontology_enum_column:
        description: test enum column with ontology values
        range: SyntheticEnumOnt
        required: True
        #ifabsent: SyntheticEnumOnt(ANIMAL)
      multivalued_column:
        description: one-to-many form
        range: integer
        required: True
        multivalued: True
        inlined_as_list: True
      multivalued_one_many_column:
        description: list form
        range: integer
        required: True
        multivalued: True
      any_type_column:
        description: needs to have type object
        range: AnyType
        required: True
      cardinality_column:
        description: check cardinality
        range: integer
        required: true
        minimum_cardinality: 1
        maximum_cardinality: 1
      class_column:
        description: test column with another class id
        range: ColumnType
        required: true
      inlined_class_column:
        description: test column with another class inlined as a struct
        range: ColumnType
        required: True
        inlined_as_list: True


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


MODEL_COLUMNS = [
    "identifier_column",
    "bool_column",
    "integer_column",
    "float_column",
    "string_column",
    "date_column",
    "datetime_column",
    "enum_column",
    "ontology_enum_column",
    "multivalued_column",
    "multivalued_one_many_column",
    "any_type_column",
    "cardinality_column",
]


@pytest.fixture(scope="module")
def np():
    """The numpy package is optional? so use fixtures and importorskip to only run tests when it's installed"""
    return pytest.importorskip("numpy", minversion="1.0", reason="Polars >= 1.0 not installed")


@pytest.fixture(scope="module")
def pl():
    """The PolaRS package is optional, so use fixtures and importorskip to only run tests when it's installed"""
    return pytest.importorskip("polars", minversion="1.0", reason="Polars >= 1.0 not installed")


@pytest.fixture(scope="module")
def pandera():
    """The pandera package is optional, so use fixtures and importorskip to only run tests when it's installed"""
    return pytest.importorskip("pandera.polars", reason="Pandera not installed")


@pytest.fixture(scope="module")
def N():
    """Number of rows in the test dataframes, 1M is enough to be real but not strain most machines."""
    return 1000000


@pytest.fixture(scope="module")
def big_synthetic_dataframe(pl, np, N):
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
                    dtype=pl.Datetime,
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
                "multivalued_column": [[1, 2, 3],] * N,
                "multivalued_one_many_column": pl.Series(np.random.choice(range(100), size=N), dtype=pl.Int64),
                "any_type_column": pl.Series([1,] * N, dtype=pl.Object),
                "cardinality_column": pl.Series(np.arange(1, N+1), dtype=pl.Int64),
                "class_column": pl.Series(np.arange(0, N), dtype=pl.Int64).cast(pl.Utf8),
            }
        )
        .with_columns(
            pl.struct(
                pl.Series(values=np.random.choice([0, 1], size=N), dtype=pl.Int64).cast(pl.Utf8).alias("id"),
                pl.Series(values=np.random.choice([0, 1], size=N), dtype=pl.Int64).alias("x"),
                pl.Series(values=np.random.choice([0, 1], size=N), dtype=pl.Int64).alias("y")
            ).alias("inlined_class_column"),
        )
    )
    # fmt: on

    return df


@pytest.fixture(scope="module")
def synthetic_schema(synthetic_flat_dataframe_model):
    return PanderaGenerator(synthetic_flat_dataframe_model)


@pytest.fixture(scope="module")
def compiled_synthetic_schema_module(synthetic_schema):
    return synthetic_schema.compile_pandera()


def test_pandera_basic_class_based(synthetic_schema):
    """
    Test generation of Pandera for classed-based mode

    This test will check the generated python, but does not include a compilation step
    """
    code = synthetic_schema.serialize()

    classes = []

    class_declaration_re = re.compile(r"class (\S+)\(")

    for item in code.splitlines():
        match = class_declaration_re.search(item)
        if match:
            classes.append(match.group(1))

    expected_classes = ["AnyType", "ColumnType", "PanderaSyntheticTable"]

    assert sorted(expected_classes) == sorted(classes)


def test_dump_schema_code(synthetic_schema):
    code = synthetic_schema.serialize()

    logger.info(f"\nGenerated Pandera model:\n{code}")

    assert all(column in code for column in MODEL_COLUMNS)


def test_get_metadata(compiled_synthetic_schema_module):
    logger.info(compiled_synthetic_schema_module.PanderaSyntheticTable.get_metadata())


def test_dump_synthetic_df(big_synthetic_dataframe):
    print(big_synthetic_dataframe)


def test_pandera_compile_basic_class_based(compiled_synthetic_schema_module, big_synthetic_dataframe):
    """
    tests compilation and validation of correct class-based schema
    """
    # raises pandera.errors.SchemaErrors, so no assert needed
    compiled_synthetic_schema_module.PanderaSyntheticTable.validate(big_synthetic_dataframe, lazy=True)


def test_pandera_validation_error_ge(pl, pandera, compiled_synthetic_schema_module, big_synthetic_dataframe):
    """
    tests ge range validation error
    """
    # fmt: off
    high_int_dataframe = (
        big_synthetic_dataframe
        .with_columns(
            pl.lit(1000, pl.Int64).alias("integer_column")
        )
    )
    # fmt: on

    with pytest.raises(pandera.errors.SchemaErrors) as e:
        compiled_synthetic_schema_module.PanderaSyntheticTable.validate(high_int_dataframe, lazy=True)

    assert "DATAFRAME_CHECK" in str(e.value)
    assert "less_than_or_equal_to(999)" in str(e.value)
    assert "'column': 'integer_column'" in str(e)


def test_pandera_validation_error_cardinality(pl, pandera, compiled_synthetic_schema_module, big_synthetic_dataframe):
    """
    tests ge range validation error
    """
    # fmt: off
    bad_cardinality_dataframe = (
        big_synthetic_dataframe
        .with_columns(
            pl.lit(1000, pl.Int64).alias("cardinality_column")
        )
    )
    # fmt: on

    with pytest.raises(pandera.errors.SchemaErrors) as e:
        compiled_synthetic_schema_module.PanderaSyntheticTable.validate(bad_cardinality_dataframe, lazy=True)

    assert "DATAFRAME_CHECK" in str(e.value)
    assert "check_cardinality_cardinality_column" in str(e.value)
    assert "'column': 'cardinality_column'" in str(e)


@pytest.mark.parametrize("bad_column", MODEL_COLUMNS)
def test_synthetic_dataframe_wrong_datatype(
    pl, pandera, compiled_synthetic_schema_module, big_synthetic_dataframe, bad_column
):
    if bad_column == "bool_column":
        bad_value = None
    else:
        bad_value = False

    # fmt: off
    error_dataframe = (
        big_synthetic_dataframe
        .with_columns(
            pl.lit(bad_value).alias(bad_column)
        )
    )
    # fmt: on

    with pytest.raises(pandera.errors.SchemaErrors) as e:
        compiled_synthetic_schema_module.PanderaSyntheticTable.validate(error_dataframe, lazy=True)

    assert "WRONG_DATATYPE" in str(e.value)
    assert f"expected column '{bad_column}' to have type" in str(e.value)


@pytest.mark.parametrize("drop_column", MODEL_COLUMNS)
def test_synthetic_dataframe_boolean_error(
    pl, pandera, compiled_synthetic_schema_module, big_synthetic_dataframe, drop_column
):

    # fmt: off
    error_dataframe = (
        big_synthetic_dataframe
        .drop(
            pl.col(drop_column)
        )
    )
    # fmt: on

    with pytest.raises(pandera.errors.SchemaErrors) as e:
        compiled_synthetic_schema_module.PanderaSyntheticTable.validate(error_dataframe, lazy=True)

    assert "COLUMN_NOT_IN_DATAFRAME" in str(e.value)
    assert f"column '{drop_column}' not in dataframe" in str(e.value)
