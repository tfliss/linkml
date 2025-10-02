import logging
import sys
from types import ModuleType

import pytest
import yaml
from linkml_runtime.utils.compile_python import compile_python

from linkml.generators.panderagen.dict_compare import deep_compare_dicts
from linkml.generators.panderagen.polars_schema.polars_schema_dataframe_generator import PolarsSchemaDataframeGenerator

_MIN_POLARS_VERSION = "1.29.0"

logger = logging.getLogger(__name__)


def apply_skip_list(skip_value: str, skip_list: list[str]) -> None:
    """Skip tests that match a string (for example schema name)"""
    for n in skip_list:
        if skip_value.startswith(n):
            pytest.skip(reason=f"Skipping test due to match on {n}")


_POLARS_SCHEMA_SKIP_LIST = [
    "test_jsonpointer",
    "test_inlined_unique_keys",
    "test_nested_key",
    "test_date_types",
    "test_array",
    "test_slot_any_of",
    "test_membership",
    "test_class_boolean",
    "test_non_standard",
    "test_equals_string",
    "test_value_presence",
    "test_cardinality-MVTrue_REQFalse",
    "test_cardinality-MVTrue_REQTrue",
    "test_cardinality-ClassNameEQ_C__SlotNameEQ_sSPACE1__TypeNameEQ_tSPACE1",
    "test_cardinality-ClassNameEQ_C__SlotNameEQ_1s__TypeNameEQ_T1",
    "test_cardinality-ClassNameEQ_C__SlotNameEQ_1s__TypeNameEQ_T1",
]

_PANDERA_SKIP_LIST = _POLARS_SCHEMA_SKIP_LIST


def generate_polars_schema(schema) -> ModuleType:
    schema_yaml = yaml.dump(schema)
    generator = PolarsSchemaDataframeGenerator(schema=schema_yaml, mergeimports=True)
    output = generator.serialize()
    logger.info(f"PolaRS Schema:\n{output}")
    mod = generator.compile_dataframe_model
    mod = compile_python(output, module_name="panderagen_polars_schema")

    return mod


def generate_polars_schema_loaded(schema) -> ModuleType:
    """The loaded polars schema always uses lists rather than dicts for inlined collections"""
    schema_yaml = yaml.dump(schema)
    generator = PolarsSchemaDataframeGenerator(schema=schema_yaml, backing_form="loaded")
    generator.template_file = "polars_schema.jinja2"
    generator.template_path = "panderagen_polars_schema"
    output = generator.serialize()
    logger.info(f"PolaRS Loaded Schema:\n{output}")
    mod = generator.compile_dataframe_model("panderagen_polars_schema_loaded")

    return mod


def generate_polars_transform(schema) -> ModuleType:
    """converts between serialized backing form using dicts and loaded using lists"""
    generate_polars_schema_loaded(schema)  # required for import dependency

    schema_yaml = yaml.dump(schema)
    generator = PolarsSchemaDataframeGenerator(schema_yaml, backing_form="transform")
    generator.template_file = "load_transformer.jinja2"
    generator.template_path = "panderagen_polars_schema"
    output = generator.serialize()
    logger.info(f"PolaRS Transform:\n{output}")
    mod = generator.compile_dataframe_model("panderagen_polars_schema_transform")

    return mod


def check_data_pandera(schema, output, target_class, object_to_validate, coerced, expected_behavior, valid):
    apply_skip_list(schema["name"], _PANDERA_SKIP_LIST)
    pl = pytest.importorskip("polars", minversion="1.0", reason="Polars >= 1.0 not installed")

    logger.info(
        f"Validating {target_class} against {object_to_validate} / {coerced} / {expected_behavior} / "
        f"{valid}\n\n{yaml.dump(schema)}\n\n{output}"
    )

    try:
        pl_schema = generate_polars_schema(schema)
        pl_schema_transform = generate_polars_transform(schema)
        mod = compile_python(output, module_name="panderagen_class_based")
        py_cls = getattr(mod, target_class)

        pl_schema_cls = getattr(pl_schema, target_class)
        dataframe_serialized_form = pl.from_dicts([object_to_validate], schema=pl_schema_cls, strict=False)

        same = deep_compare_dicts(object_to_validate, dataframe_serialized_form.to_dicts()[0])
        if not same and valid:
            assert same, f"PolaRS schema did not match input object for {schema['name']}"
        elif not same and not valid:
            logger.info("PolaRS schema did not load invalid object to validate properly")

        logger.info(dataframe_serialized_form)

        xform_cls = getattr(pl_schema_transform, target_class)
        xform = xform_cls()
        dataframe_to_validate = xform.load(dataframe_serialized_form)

        py_cls.validate(dataframe_to_validate, lazy=True)
    except Exception as e:
        logger.info(f"Schema Name: {schema['name']}")
        if valid:
            logger.info(output)
            raise e
    finally:
        sys.modules.pop("panderagen_polars_schema", None)
        sys.modules.pop("panderagen_class_based", None)


def check_data_polars_schema(schema, output, target_class, object_to_validate, coerced, expected_behavior, valid):
    """
    Note: this test passes even if invalid objects are loaded, because the schema is not a validator.
    """
    apply_skip_list(schema["name"], _POLARS_SCHEMA_SKIP_LIST)
    pl = pytest.importorskip("polars", minversion=_MIN_POLARS_VERSION, reason="Polars >= 1.0 not installed")

    try:
        logger.info(
            f"Validating {target_class} against {object_to_validate} / {coerced} / {expected_behavior} / "
            f"{valid}\n\n{yaml.dump(schema)}\n\n{output}"
        )

        logger.info(f"Behavior: {expected_behavior}")
        logger.info(f"Valid: {valid}")
        logger.info(f"Expected: {object_to_validate}")

        mod = compile_python(output)  # , module_name="test_polars_schema")
        py_cls = getattr(mod, target_class)

        dataframe_to_validate = pl.from_dicts([object_to_validate], schema=py_cls)

        same = deep_compare_dicts(object_to_validate, dataframe_to_validate.to_dicts()[0])

        logger.info(f"Actual: {dataframe_to_validate.to_dicts()[0]}")
        logger.info(f"Same: {same}")

        if same and not valid:
            logger.warning("PolaRS schema accepted an invalid object. Note the schema is not a full validator.")
        assert same
    except Exception as e:
        logger.info("Actual: EXCEPTION")
        logger.info("Same: N/A")
        logger.info(f"Schema Name: {schema['name']}")
        if valid:
            raise e
