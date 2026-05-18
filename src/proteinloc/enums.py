from enum import Enum

class ModelName(str, Enum):
    esm_300 = "esm_300"
    esm_600 = "esm_600"
    prost = "prost"

class OutputFormat(str, Enum):
    table = "table"
    json = "json"
    csv = "csv"