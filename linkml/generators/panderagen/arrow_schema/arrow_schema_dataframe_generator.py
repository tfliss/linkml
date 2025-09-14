import logging
from dataclasses import dataclass

from ..dataframe_generator import DataframeGenerator

logger = logging.getLogger(__name__)


@dataclass
class ArrowSchemaDataframeGenerator(DataframeGenerator):
    """
    Generates PyArrow schema classes from a LinkML schema.
    """

    TEMPLATE_DIRECTORY = "panderagen_arrow_schema"

    TYPE_MAP = {
        "xsd:string": "pa.string",
        "xsd:integer": "pa.int64",
        "xsd:int": "pa.int32",
        "xsd:float": "pa.float32",
        "xsd:double": "pa.float64",
        "xsd:boolean": "pa.boolean",
        "xsd:dateTime": "pa.timestamp",
        "xsd:date": "pa.date64",
        "xsd:time": "pa.time64",
        "xsd:anyURI": "pa.string",
        "xsd:decimal": "pa.decimal128",
    }

    @staticmethod
    def make_multivalued(range: str) -> str:
        if range == "Struct":
            return "pa.list_"
        return f"List[{range}]"
