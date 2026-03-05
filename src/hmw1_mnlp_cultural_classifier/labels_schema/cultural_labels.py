from dataclasses import dataclass
from typing import Dict, List


class CulturalLabels:
    """
    Canonical label schema for cultural classification.
    """
    id_to_name: Dict[int, str] = {
        0: "cultural agnostic",
        1: "cultural representative",
        2: "cultural exclusive"
    }

    @property
    def name_to_id(self) -> Dict[str, int]:
        return {v: k for k, v in self.id_to_name.items()}

    @property
    def num_labels(self) -> int:
        return len(self.id_to_name)

    @property
    def names(self) -> List[str]:
        return list(self.name_to_id.keys())
