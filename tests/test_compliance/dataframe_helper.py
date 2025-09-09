import logging

import pytest
import yaml
from linkml_runtime.utils.compile_python import compile_python

from linkml.generators.panderagen.dict_compare import deep_compare_dicts

logger = logging.getLogger(__name__)


def check_data_pandera(schema, output, target_class, object_to_validate, coerced, expected_behavior, valid):
    pl = pytest.importorskip("polars", minversion="1.0", reason="Polars >= 1.0 not installed")

    try:
        mod = compile_python(output)
        py_cls = getattr(mod, target_class)

        logger.info(
            f"Validating {target_class} against {object_to_validate} / {coerced} / {expected_behavior} / "
            f"{valid}\n\n{yaml.dump(schema)}\n\n{output}"
        )

        dataframe_to_validate = pl.DataFrame([object_to_validate])

        try:
            schema_name = schema.get("name", "")
            polars_schema = py_cls.generate_polars_schema(object_to_validate, parser=True)

            if schema_name.startswith("test_date_types") or schema_name.startswith("test_enum_alias"):
                dataframe_to_validate = pl.DataFrame(object_to_validate, schema=polars_schema, strict=False)
            elif dataframe_to_validate.item() is None:
                dataframe_to_validate = pl.DataFrame(object_to_validate, schema=polars_schema, strict=False)
        except Exception:
            pass

        logger.info(dataframe_to_validate)
        py_cls.validate(dataframe_to_validate, lazy=True)
    except Exception as e:
        if valid:
            raise e


def check_data_polars_schema(schema, output, target_class, object_to_validate, coerced, expected_behavior, valid):
    """
    Note: this test passes even if invalid objects are loaded, because the schema is not a validator.
    """

    for n in [
        "test_jsonpointer",
        "test_inlined",
        "test_unique_keys",
        "test_nested_key",
        "test_date_types",
        "test_array",
        "test_slot_any_of",
        "test_inlined-INLFalse_IALFalse_MVTrue_FKTrue",
        "test_enum_hierarchy",
        "test_cardinality-ClassNameEQ_C__SlotNameEQ_sSPACE1__TypeNameEQ_tSPACE1",
        "test_cardinality-ClassNameEQ_C__SlotNameEQ_1s__TypeNameEQ_T1",
        "",
    ]:
        if schema["name"].startswith(n):
            pytest.skip(reason="Not implemented")
        else:
            logger.info(f"Checking schema {schema['name']}")
    pl = pytest.importorskip("polars", minversion="1.0", reason="Polars >= 1.0 not installed")

    try:
        logger.info(
            f"Validating {target_class} against {object_to_validate} / {coerced} / {expected_behavior} / "
            f"{valid}\n\n{yaml.dump(schema)}\n\n{output}"
        )

        logger.info(f"Behavior: {expected_behavior}")
        logger.info(f"Valid: {valid}")
        logger.info(f"Expected: {object_to_validate}")

        mod = compile_python(output)
        py_cls = getattr(mod, target_class)

        dataframe_to_validate = pl.from_dicts([object_to_validate], schema=py_cls)

        # logger.info(dataframe_to_validate)

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
