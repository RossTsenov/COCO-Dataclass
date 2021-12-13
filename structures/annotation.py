from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

from structures.types import Annotation, compare_dict_structure


@dataclass
class RLE:
    """Dataclass for Run Length Encoding."""
    counts: List[float] = field(default_factory=list)
    size: List[float] = field(default_factory=list)

    @classmethod
    def from_dict(cls, rle_dict: Dict[str, List[float]], ignore_index = True):
        """Generates Run Length Encoding dataclass from dictionary.

        Args:
            rle_dict (Dict[str, Any]): Dictionary objects that has Run Length Encoding structure.
            ignore_extra_keys (bool, optional): Ignore the fact dictionary has more fields than specified in dataset. Defaults to True.

        Returns:
            Run Length Encoding: Object generated from dictionary.
        """
        compare_dict_structure(rle_dict, cls, ignore_index)
        return cls(**rle_dict)


@dataclass
class SegmentInfo:
    """Dataclass for Segment Info for the Panoptic Segmentation Annotation."""
    id: int
    category_id: int
    area: int = 0
    bbox: List[int] = field(default_factory=list)
    iscrowd: int = 0

    @classmethod
    def from_dict(cls, segment_info_dict: Dict[str, Any], ignore_index = True):
        """Generates Segment Info for the Panoptic Segmentation Annotation dataclass from dictionary.

        Args:
            segment_info_dict (Dict[str, Any]): Dictionary objects that has COCO Segment Info structure.
            ignore_extra_keys (bool, optional): Ignore the fact dictionary has more fields than specified in dataset. Defaults to True.

        Returns:
            Segment Info: Object generated from dictionary.
        """
        compare_dict_structure(segment_info_dict, cls, ignore_index)
        return cls(**segment_info_dict)


@dataclass
class ObjectDetectionAnnotation:
    """Dataclass that mimics Object Detection Annotation structure of COCO dataset.
    Follows `Annotation`, `Indexed` and `Categorized` custom Protocols.
    """
    id: int
    image_id: int
    category_id: int
    segmentation: Union[List[List[float]], RLE]
    area: float = 0
    bbox: List[float] = field(default_factory=list)
    iscrowd: int = 0

    @classmethod
    def from_dict(cls, annotation_dict: Dict[str, Any], ignore_index = True):
        """Generates Object Detection Annotation dataclass from dictionary.

        Args:
            annotation_dict (Dict[str, Any]): Dictionary objects that has COCO Object Detection Annotation structure.
            ignore_extra_keys (bool, optional): Ignore the fact dictionary has more fields than specified in dataset. Defaults to True.

        Returns:
            Object Detection Annotation: Object generated from dictionary.
        """
        compare_dict_structure(annotation_dict, cls, ignore_index)
        return cls(
            id = annotation_dict['id'],
            image_id = annotation_dict['image_id'],
            category_id = annotation_dict['category_id'],
            segmentation = (
                annotation_dict['segmentation']
                if not isinstance(annotation_dict['segmentation'], dict)
                else RLE.from_dict(annotation_dict['segmentation'], ignore_index)
            ),
            area = annotation_dict['area'],
            bbox = annotation_dict['bbox'],
            iscrowd = annotation_dict['iscrowd'],
        )


@dataclass
class KeypointDetectionAnnotation(ObjectDetectionAnnotation):
    """Dataclass that mimics Keypoint Detection Annotation structure of COCO dataset.
    Follows `Annotation`, `Indexed` and `Categorized` custom Protocols.
    """
    keypoints: List[int] = field(default_factory=list)
    num_keypoints: int = 0

    @classmethod
    def from_dict(cls, annotation_dict: Dict[str, Any], ignore_index = True):
        """Generates Keypoint Detection Annotation dataclass from dictionary.

        Args:
            annotation_dict (Dict[str, Any]): Dictionary objects that has COCO Keypoint Detection Annotation structure.
            ignore_extra_keys (bool, optional): Ignore the fact dictionary has more fields than specified in dataset. Defaults to True.

        Returns:
            Keypoint Detection Annotation: Object generated from dictionary.
        """
        compare_dict_structure(annotation_dict, cls, ignore_index)
        return cls(
            id = annotation_dict['id'],
            image_id = annotation_dict['image_id'],
            category_id = annotation_dict['category_id'],
            keypoints = annotation_dict['keypoints'],
            num_keypoints = annotation_dict['num_keypoints'],
            segmentation = (
                annotation_dict['segmentation']
                if not isinstance(annotation_dict['segmentation'], dict)
                else RLE.from_dict(annotation_dict['segmentation'], ignore_index)
            ),
            area = annotation_dict['area'],
            bbox = annotation_dict['bbox'],
            iscrowd = annotation_dict['iscrowd'],
        )


@dataclass
class StuffSegmentationAnnotation(ObjectDetectionAnnotation):
    """Dataclass that mimics Stuff Segmentation Annotation structure of COCO dataset.
    Follows `Annotation`, `Indexed` and `Categorized` custom Protocols.
    """


@dataclass
class PanopticSegmentationAnnotation:
    """Dataclass that mimics Panoptic Segmentation Annotation structure of COCO dataset.
    Follows `Annotation` custom Protocol.
    """
    image_id: int
    file_name: str = ""
    segments_info: List[SegmentInfo] = field(default_factory=list)

    @classmethod
    def from_dict(cls, annotation_dict: Dict[str, Any], ignore_index = True):
        """Generates Panoptic Segmentation Annotation dataclass from dictionary.

        Args:
            annotation_dict (Dict[str, Any]): Dictionary objects that has COCO Panoptic Segmentation Annotation structure.
            ignore_extra_keys (bool, optional): Ignore the fact dictionary has more fields than specified in dataset. Defaults to True.

        Returns:
            Panoptic Segmentation Annotation: Object generated from dictionary.
        """
        compare_dict_structure(annotation_dict, cls, ignore_index)
        return cls(
            image_id = annotation_dict['image_id'],
            file_name = annotation_dict['file_name'],
            segments_info = SegmentInfo.from_dict(annotation_dict['segments_info'], ignore_index),
        )


@dataclass
class ImageCaptioningAnnotation:
    """Dataclass that mimics Image Captioning Annotation structure of COCO dataset.
    Follows `Annotation` and `Indexed` custom Protocols.
    """
    id: int
    image_id: int
    caption: str = ""

    @classmethod
    def from_dict(cls, annotation_dict: Dict[str, Any], ignore_index = True):
        """Generates Image Captioning Annotation dataclass from dictionary.

        Args:
            annotation_dict (Dict[str, Any]): Dictionary objects that has COCO Image Captioning Annotation structure.
            ignore_extra_keys (bool, optional): Ignore the fact dictionary has more fields than specified in dataset. Defaults to True.

        Returns:
            Image Captioning Annotation: Object generated from dictionary.
        """
        compare_dict_structure(annotation_dict, cls, ignore_index)
        return cls(**annotation_dict)


@dataclass
class DensePoseAnnotation:
    """Dataclass that mimics Dense Pose Annotation structure of COCO dataset.
    Follows `Annotation`, `Indexed` and `Categorized` custom Protocols.
    """
    id: int
    image_id: int
    category_id: int
    is_crowd: int = 0
    area: int = 0
    bbox: List[int] = field(default_factory=list)
    dp_I: List[float] = field(default_factory=list)
    dp_U: List[float] = field(default_factory=list)
    dp_V: List[float] = field(default_factory=list)
    dp_x: List[float] = field(default_factory=list)
    dp_y: List[float] = field(default_factory=list)
    dp_masks: List[RLE] = field(default_factory=list)

    @classmethod
    def from_dict(cls, annotation_dict: Dict[str, Any], ignore_index = True):
        """Generates Dense Pose Annotation dataclass from dictionary.

        Args:
            annotation_dict (Dict[str, Any]): Dictionary objects that has COCO Dense Pose Annotation structure.
            ignore_extra_keys (bool, optional): Ignore the fact dictionary has more fields than specified in dataset. Defaults to True.

        Returns:
            Dense Pose Annotation: Object generated from dictionary.
        """
        compare_dict_structure(annotation_dict, cls, ignore_index)
        return cls(
            id = annotation_dict['id'],
            image_id = annotation_dict['image_id'],
            category_id = annotation_dict['category_id'],
            is_crowd = annotation_dict['is_crowd'],
            area = annotation_dict['area'],
            bbox = annotation_dict['bbox'],
            dp_I = annotation_dict['dp_I'],
            dp_U = annotation_dict['dp_U'],
            dp_V = annotation_dict['dp_V'],
            dp_x = annotation_dict['dp_x'],
            dp_y = annotation_dict['dp_y'],
            dp_masks = [RLE.from_dict(mask, ignore_index) for mask in annotation_dict['dp_masks']],
        )



DICT_TO_ANNOTATION_MAP = {
    "object_detection" : ["id", "image_id", "category_id", "segmentation", "area", "bbox", "iscrowd"],
    "keypoint_detection" : ["id", "image_id", "category_id", "segmentation", "area", "bbox", "iscrowd", "keypoints", "skeleton"],
    "panoptic_segmentation" : ["image_id", "file_name", "segments_info"],
    "image_captioning" : ["id", "image_id", "caption"],
    "dense_pose" : ["id", "image_id", "category_id", "is_crowd", "area", "bbox", "dp_I", "dp_U", "dp_V", "dp_x", "dp_y", "dp_masks"],
}


def dict_to_annotation(annotation_dict: Dict[str, Any], ignore_extra_keys = True) -> Annotation:
    """Calls specific Category object constructor based on the structure of the `annotation_dict`.

    Args:
        annotation_dict (Dict[str, Any]): One of COCO Annotation dictionaries.
        ignore_extra_keys (bool, optional): Ignore the fact dictionary has more fields than specified in dataset. Defaults to True.

    Raises:
        ValueError: If `annotation_dict` has unspecified structure.

    Returns:
        Annotation: Dataclass category generated from the `annotation_dict`.
    """
    if set(DICT_TO_ANNOTATION_MAP['object_detection']).issubset(annotation_dict.keys()):
        return ObjectDetectionAnnotation.from_dict(annotation_dict, ignore_extra_keys)

    elif set(DICT_TO_ANNOTATION_MAP['keypoint_detection']).issubset(annotation_dict.keys()):
        return KeypointDetectionAnnotation.from_dict(annotation_dict, ignore_extra_keys)

    elif set(DICT_TO_ANNOTATION_MAP['panoptic_segmentation']).issubset(annotation_dict.keys()):
        return PanopticSegmentationAnnotation.from_dict(annotation_dict, ignore_extra_keys)

    elif set(DICT_TO_ANNOTATION_MAP['image_captioning']).issubset(annotation_dict.keys()):
        return ImageCaptioningAnnotation.from_dict(annotation_dict, ignore_extra_keys)

    elif set(DICT_TO_ANNOTATION_MAP['dense_pose']).issubset(annotation_dict.keys()):
        return DensePoseAnnotation.from_dict(annotation_dict, ignore_extra_keys)

    raise ValueError(
        "Unexpected annotation structure. Consider manually creating COCO dataset."
        "\nAnd extending one of existing objects or create new following one of the Protocols structure."
    )
