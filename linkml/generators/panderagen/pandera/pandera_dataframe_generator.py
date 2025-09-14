import logging
from dataclasses import dataclass
from enum import Enum

from ..class_generator_mixin import ClassGeneratorMixin
from ..dataframe_generator import DataframeGenerator
from ..enum_generator_mixin import EnumGeneratorMixin
from .slot_generator_mixin_pandera import SlotGeneratorMixinPandera

logger = logging.getLogger(__name__)


class TemplateEnum(Enum):
    CLASS_BASED = "panderagen_class_based"
    OBJECT_BASED = "panderagen_object_based"
    POLARS_SCHEMA = "polars_schema"
    PYARROW_SCHEMA = "pyarrow_schema"


@dataclass
class PanderaDataframeGenerator(DataframeGenerator, EnumGeneratorMixin, ClassGeneratorMixin, SlotGeneratorMixinPandera):
    """
    Generates Pandera python classes from a LinkML schema.

    Status: incompletely implemented

    Two styles are supported:

    - class-based
    - schema-based (not implemented)
    """

    TEMPLATE_DIRECTORY = "panderagen_class_based"

    # Pandera-specific type mapping
    TYPE_MAP = {
        "xsd:string": "str",
        "xsd:integer": "int",
        "xsd:int": "int",
        "xsd:float": "float",
        "xsd:double": "float",
        "xsd:boolean": "bool",
        "xsd:dateTime": "DateTime()",
        "xsd:date": "Date",
        "xsd:time": "Time",
        "xsd:anyURI": "str",
        "xsd:decimal": "float",
    }

    # ObjectVars
    inline_validator_mixin: bool = False
    coerce: bool = False

    @staticmethod
    def make_multivalued(range: str) -> str:
        return "List"
