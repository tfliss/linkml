import logging

from linkml.generators.panderagen.linkml_polars_transformer import LinkmlPolarsTransformer

logger = logging.getLogger(__name__)


def test_polars_transform(
    compiled_synthetic_schema_module,
    compiled_synthetic_schema_loaded,
    compiled_synthetic_schema_transform,
    big_synthetic_dataframe,
):
    logger.info(big_synthetic_dataframe)

    xform = compiled_synthetic_schema_transform.XXXTransformer(
        compiled_synthetic_schema_loaded.PanderaSyntheticTableDict
    )

    schema = compiled_synthetic_schema_loaded.PanderaSyntheticTableDict
    col = "inlined_class_column"
    op = LinkmlPolarsTransformer(schema)
    op.transform_collection_dict(
        big_synthetic_dataframe, col, reference_cls=compiled_synthetic_schema_loaded.ColumnTypeStruct
    )

    loaded = xform.load_PanderaSyntheticTable(big_synthetic_dataframe)

    logger.info(loaded)

    assert compiled_synthetic_schema_module is not None
    assert compiled_synthetic_schema_loaded is not None
    assert compiled_synthetic_schema_transform is not None
