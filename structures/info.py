from dataclasses import dataclass
from typing import Any, Dict

from structures.types import compare_dict_structure


@dataclass
class Info:
    """Dataclass that mimics Info structure of COCO dataset."""
    year: int = 0
    version: str = ""
    description: str = ""
    contributor: str = ""
    url: str = ""
    date_created: str = ""

    @classmethod
    def from_dict(cls, info_dict: Dict[str, Any], ignore_extra_keys = True):
        """Generates Info dataclass from dictionary.

        Args:
            info_dict (Dict[str, Any]): Dictionary objects that has COCO Info structure.
            ignore_extra_keys (bool, optional): Ignore the fact dictionary has more fields than specified in dataset. Defaults to True.

        Returns:
            Info: Object generated from dictionary.
        """
        compare_dict_structure(info_dict, cls, ignore_extra_keys)
        return cls(**info_dict)
