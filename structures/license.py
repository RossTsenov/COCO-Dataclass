from dataclasses import dataclass
from typing import Any, Dict

from structures.types import compare_dict_structure


@dataclass
class License:
    """Dataclass that mimics License structure of COCO dataset."""
    id: int
    name: str = ""
    url: str = ""

    @classmethod
    def from_dict(cls, license_dict: Dict[str, Any], ignore_extra_keys = True):
        """Generates License dataclass from dictionary.

        Args:
            license_dict (Dict[str, Any]): Dictionary objects that has COCO License structure.
            ignore_extra_keys (bool, optional): Ignore the fact dictionary has more fields than specified in dataset. Defaults to True.

        Returns:
            License: Object generated from dictionary.
        """
        compare_dict_structure(license_dict, cls, ignore_extra_keys)
        return cls(**license_dict)
