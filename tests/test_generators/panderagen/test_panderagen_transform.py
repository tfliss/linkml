import logging

logger = logging.getLogger(__name__)


def test_polars_transform(
    compiled_synthetic_schema_module,
    compiled_synthetic_schema_loaded,
    compiled_synthetic_schema_transform,
    big_synthetic_dataframe_serialized,
):
    logger.info(big_synthetic_dataframe_serialized)

    xform = compiled_synthetic_schema_transform.PanderaSyntheticTable()
    loaded = xform.load(big_synthetic_dataframe_serialized)

    logger.info(loaded)

    assert compiled_synthetic_schema_module is not None
    assert compiled_synthetic_schema_loaded is not None
    assert compiled_synthetic_schema_transform is not None


def test_validate_transformed_df(
    compiled_synthetic_pandera_schema_module,
    compiled_synthetic_schema_transform,
    big_synthetic_dataframe_serialized,
):
    xform = compiled_synthetic_schema_transform.PanderaSyntheticTable()
    loaded = xform.load(big_synthetic_dataframe_serialized)

    validated = compiled_synthetic_pandera_schema_module.PanderaSyntheticTable.validate(loaded)

    assert validated is not None
