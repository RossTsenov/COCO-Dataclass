from dataclasses import dataclass, field
from typing import Any, Dict, List

from structures.types import Category, compare_dict_structure


@dataclass
class ObjectDetectionCategory:
    """Dataclass that mimics Object Detection Category structure of COCO dataset.
    Follows `Category` custom Protocol.
    """
    id: int
    name: str = ""
    supercategory: str = ""

    @classmethod
    def from_dict(cls, category_dict: Dict[str, Any], ignore_extra_keys = True):
        """Generates Object Detection Category dataclass from dictionary.

        Args:
            category_dict (Dict[str, Any]): Dictionary objects that has COCO Object Detection Category structure.
            ignore_extra_keys (bool, optional): Ignore the fact dictionary has more fields than specified in dataset. Defaults to True.

        Returns:
            Object Detection Category: Object generated from dictionary.
        """
        compare_dict_structure(category_dict, cls, ignore_extra_keys)
        return cls(**category_dict)


@dataclass
class KeypointDetectionCategory(ObjectDetectionCategory):
    """Dataclass that mimics Keypoint Detection Category structure of COCO dataset.
    Follows `Category` custom Protocol.
    """
    keypoints: List[str] = field(default_factory=list)
    skeleton: List[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, category_dict: Dict[str, Any], ignore_extra_keys = True):
        """Generates Keypoint Detection Category dataclass from dictionary.

        Args:
            category_dict (Dict[str, Any]): Dictionary objects that has COCO Keypoint Detection Category structure.
            ignore_extra_keys (bool, optional): Ignore the fact dictionary has more fields than specified in dataset. Defaults to True.

        Returns:
            Keypoint Detection Category: Object generated from dictionary.
        """
        compare_dict_structure(category_dict, cls, ignore_extra_keys)
        return cls(**category_dict)


@dataclass
class PanopticSegmentationCategory(ObjectDetectionCategory):
    """Dataclass that mimics Panoptic Segmentation Category structure of COCO dataset.
    Follows `Category` custom Protocol.
    """
    isthing: int = 0
    color: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, category_dict: Dict[str, Any], ignore_extra_keys = True):
        """Generates Panoptic Segmentation Category dataclass from dictionary.

        Args:
            category_dict (Dict[str, Any]): Dictionary objects that has COCO Panoptic Segmentation Category structure.
            ignore_extra_keys (bool, optional): Ignore the fact dictionary has more fields than specified in dataset. Defaults to True.

        Returns:
            Panoptic Segmentation Category: Object generated from dictionary.
        """
        compare_dict_structure(category_dict, cls, ignore_extra_keys)
        return cls(**category_dict)



DICT_TO_CATEGORY_MAP = {
    "object_detection" : ["id", "name", "supercategory"], # same for stuff segmentation.
    "keypoint_detection" : ["id", "name", "supercategory", "keypoints", "skeleton"],
    "panoptic_segmentation" : ["id", "name", "supercategory", "isthing", "color"],
}


def dict_to_category(category_dict: Dict[str, Any], ignore_extra_keys = True) -> Category:
    """Calls specific Category object constructor based on the structure of the `category_dict`.

    Args:
        category_dict (Dict[str, Any]): One of COCO Category dictionaries.
        ignore_extra_keys (bool, optional): Ignore the fact dictionary has more fields than specified in dataset. Defaults to True.

    Raises:
        ValueError: If `category_dict` has unspecified structure.

    Returns:
        Category: Dataclass category generated from the `category_dict`.
    """
    if set(DICT_TO_CATEGORY_MAP['object_detection']).issubset(category_dict.keys()):
        return ObjectDetectionCategory.from_dict(category_dict, ignore_extra_keys)

    elif set(DICT_TO_CATEGORY_MAP['keypoint_detection']).issubset(category_dict.keys()):
        return KeypointDetectionCategory.from_dict(category_dict, ignore_extra_keys)

    elif set(DICT_TO_CATEGORY_MAP['panoptic_segmentation']).issubset(category_dict.keys()):
        return PanopticSegmentationCategory.from_dict(category_dict, ignore_extra_keys)

    raise ValueError(
        "Unexpected category structure. Consider manually creating COCO dataset."
        "\nAnd extending one of existing objects or create new following one of the Protocol structure."
    )
