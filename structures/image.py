from dataclasses import dataclass
from typing import Any, Dict

from structures.types import compare_dict_structure


@dataclass
class Image:
    """Dataclass that mimics Image structure of COCO dataset."""
    id: int
    width: int
    height: int
    file_name: str
    license: int = 0
    flickr_url: str = ""
    coco_url: str = ""
    date_captured: str = ""

    @classmethod
    def from_dict(cls, image_dict: Dict[str, Any], ignore_extra_keys = True):
        """Generates Image dataclass from dictionary.

        Args:
            info_dict (Dict[str, Any]): Dictionary objects that has COCO Image structure.
            ignore_extra_keys (bool, optional): Ignore the fact dictionary has more fields than specified in dataset. Defaults to True.

        Returns:
            Image: Object generated from dictionary.
        """
        compare_dict_structure(image_dict, cls, ignore_extra_keys)
        return cls(**image_dict)
