from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class CulturalTextBuilder:
    """
    Builds a single input text from a dataset row.
    """
    fields: List[str] = None
    sep: str = " [SEP] "

    def __post_init__(self):
        if self.fields is None:
            object.__setattr__(self, "fields", ["name", "description", "category", "subcategory"])

    def build(self, row: Dict) -> str:
        parts = []
        for f in self.fields:
            v = row.get(f, "")
            if v:
                parts.append(str(v))
        return self.sep.join(parts)
