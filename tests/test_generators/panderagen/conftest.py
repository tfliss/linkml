import logging
from pathlib import PurePosixPath

import pytest

# Load optional dependencies using importorskip to avoid pytest collection errors
pl = pytest.importorskip("polars", minversion="1.0", reason="Polars >= 1.0 not installed")
np = pytest.importorskip("numpy", reason="NumPY not installed")


# These depend on PolaRS and Numpy so need to be after importerskip
from linkml.generators.panderagen import PanderaDataframeGenerator  # noqa: E402
from linkml.generators.panderagen.dataframe_generator import DataframeGenerator  # noqa: E402
from linkml.generators.panderagen.panderagen import PANDERA_GROUP  # noqa: E402

logger = logging.getLogger(__file__)


@pytest.fixture(scope="module")
def N():
    """Number of rows in the test dataframes, 10K is enough to be real but not strain most machines."""
    return 10000


@pytest.fixture(scope="module")
def synthetic_model_path():
    return PurePosixPath(__file__).parent / "input" / "synthetic_model.yaml"


@pytest.fixture(scope="module")
def synthetic_flat_dataframe_model(synthetic_model_path):
    with open(synthetic_model_path) as f:
        return f.read()


@pytest.fixture(scope="module")
def compiled_modules(synthetic_flat_dataframe_model):
    compiled_modules = DataframeGenerator.compile_package_from_specification(
        PANDERA_GROUP, "test_package", synthetic_flat_dataframe_model
    )

    yield compiled_modules

    DataframeGenerator.cleanup_package("test_package")


@pytest.fixture(scope="module")
def compiled_synthetic_schema_module(compiled_modules):
    return compiled_modules["panderagen_polars_schema"]


@pytest.fixture(scope="module")
def compiled_synthetic_schema_loaded(compiled_modules):
    return compiled_modules["panderagen_polars_schema_loaded"]


@pytest.fixture(scope="module")
def compiled_synthetic_schema_transform(compiled_modules):
    return compiled_modules["panderagen_polars_schema_transform"]


@pytest.fixture(scope="module")
def synthetic_pandera_schema(synthetic_flat_dataframe_model):
    return PanderaDataframeGenerator(synthetic_flat_dataframe_model)


@pytest.fixture(scope="module")
def compiled_synthetic_pandera_schema_module(compiled_modules):
    """The pandera schema using the loaded backing form"""
    return compiled_modules["panderagen_schema_loaded"]


@pytest.fixture(scope="module")
def compiled_synthetic_pandera_schema_module_serialized(compiled_modules):
    return compiled_modules["panderagen_class_based"]


@pytest.fixture(scope="module")
def column_type_instances():
    """valid NestedRecord instances that can be used in tests"""
    return [
        {
            "code": "thing_one",
            "metric": 1111,
            "score": 2222,
        },
        {
            "code": "thing_two",
            "metric": 3333,
            "score": 4444,
        },
    ]


@pytest.fixture(scope="module")
def invalid_column_type_instances():
    """invalid (float values) ColumnType instances that can trigger failures."""
    return [
        {
            "code": "thing_one",
            "metric": 1111.1,
            "score": 2222.2,
        },
        {
            "code": "thing_two",
            "metric": 3333.3,
            "score": 4444.4,
        },
    ]


@pytest.fixture(scope="module")
def valid_inlined_dict_column_expression(column_type_instances):
    """synthetic data that conforms to the inlined_mapp_column schema
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
    """synthetic data that conforms to the inlined_map_column schema
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
def valid_nested_simple_dict_column_expression(column_type_instances):
    """synthetic data that conforms to the nested_simple_dict_column schema."""
    return {"A": column_type_instances, "B": column_type_instances, "C": column_type_instances}


@pytest.fixture(scope="module")
def valid_double_nested_simple_dict_column_expression(valid_nested_simple_dict_column_expression):
    """synthetic data that conforms to the doubld nested simple dict column schema."""
    return {
        "X": valid_nested_simple_dict_column_expression,
        "Y": valid_nested_simple_dict_column_expression,
        "Z": valid_nested_simple_dict_column_expression,
    }


@pytest.fixture(scope="module")
def invalid_simple_dict_column_expression():
    """synthetic data with float values that does not conform to the inlined_simple_dict_column schema."""
    return {"A": 1.1, "B": 2.2, "C": 3.3}


@pytest.fixture(scope="module")
def big_synthetic_dataframe_serialized(
    N,
    column_type_instances,
    valid_inlined_dict_column_expression,
    valid_simple_dict_column_expression,
    compiled_synthetic_schema_module,
):
    """
    Construct a reasonably sized dataframe that complies with the DataframeRow model.
    Uses 'serialized' backing form including inefficient dict collections.
    """
    DemoEnum = compiled_synthetic_schema_module.DemoEnum
    DemoOntologyEnum = compiled_synthetic_schema_module.DemoOntologyEnum

    # fmt: off
    df = (
        pl.DataFrame(
            {
                "id_column": pl.Series(np.arange(0, N), dtype=pl.Int64),
                "flag_column": pl.Series(np.random.choice([True, False], size=N), dtype=pl.Boolean),
                "count_column": pl.Series(np.random.choice(range(100), size=N), dtype=pl.Int64),
                "ratio_column": pl.Series(np.random.choice([1.0, 2.0, 3.0], size=N), dtype=pl.Float64),
                "label_column": np.random.choice(["this", "that"], size=N),
                "event_date_column": pl.Series(
                    np.random.choice(["2021-03-27", "2021-03-28"], size=N),
                    dtype=pl.Date,
                    strict=False
                ),
                "event_datetime_column": pl.Series(
                    np.random.choice(["2021-03-27T03:00:00", "2021-03-28T03:00:00"], size=N),
                    dtype=pl.Datetime(time_unit='us', time_zone=None),
                    strict=False
                ),
                  "enum_column": pl.Series(
                      np.random.choice(DemoEnum.categories, size=N),
                      dtype=DemoEnum,
                      strict=False
                  ),
                  "ontology_enum_column": pl.Series(
                      np.random.choice(DemoOntologyEnum.categories, size=N),
                      dtype=DemoOntologyEnum,
                      strict=False
                  ),
                "multivalued_column": [[1, 2, 3]] * N,
                "any_type_column": [1] * N,
                "singular_column": np.arange(1, N+1),
                "nested_object_column": [ column_type_instances[0] ] * N,
                "foreign_key_column": [ "thing_one" ] * N,
                "inlined_simple_dict_column": [valid_simple_dict_column_expression] * N,
                #"inlined_nested_simple_dict_column": [ valid_nested_simple_dict_column_expression] * N,
                #"double_nested_simple_dict_column": [valid_double_nested_simple_dict_column_expression] * N,
                "inlined_list_column": [ column_type_instances ] * N,
                "inlined_map_column": [ valid_inlined_dict_column_expression ] * N, # is multivalued collection dict
            },
            schema=compiled_synthetic_schema_module.DataframeRowFull
        )
    )
    # fmt: on

    logger.info(df)

    return df


@pytest.fixture(scope="module")
def big_synthetic_dataframe(
    big_synthetic_dataframe_serialized,
    compiled_synthetic_schema_transform,
):
    """Synthetic dataframe with inefficient inline forms converted to lists"""
    dict_to_list_transform = compiled_synthetic_schema_transform.DataframeRowFull()
    return dict_to_list_transform.load(big_synthetic_dataframe_serialized)
