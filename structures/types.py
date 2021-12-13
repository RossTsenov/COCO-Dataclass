
from typing import List, Tuple, TypeVar, Any, Dict, Protocol, Union, runtime_checkable
from dataclasses import fields

T = TypeVar('T')

@runtime_checkable
class Category(Protocol):
    id: int
    name: str
    supercategory: str

@runtime_checkable
class Annotation(Protocol):
    image_id: int

@runtime_checkable
class Indexed(Protocol):
    id: int

@runtime_checkable
class Categorized(Protocol):
    category_id: int


def compare_dict_structure(dictionatry: Dict[str, Any], obj: Any, ignore_extra_keys = True):
    """Compares if dictionary follows the structure of the dataclass.

    Args:
        dictionatry (Dict[str, Any]): Dictionary which structure will be compared
        obj (Any): dataclass that will be comapred with dict
        ignore_extra_keys (bool, optional): Ignore the fact dictionary has more fields than specified in dataset. Defaults to True.

    Raises:
        ValueError: If dictionary is missing some fields
        ValueError: If dictionary has extraf fields, will be raised only if `ignore_extra_keys` is False.
    """
    missed_keys = {field.name for field in fields(obj)}.difference(dictionatry.keys())
    extra_keys = set(dictionatry.keys()).difference(field.name for field in fields(obj))

    if missed_keys:
        raise ValueError(
            f'It seems that dataset field is missing required keys {missed_keys}.'
            '\nCheck if dataset follows official COCO dataset structure.'
        )

    elif extra_keys and not ignore_extra_keys:
        raise ValueError(
            f'It seems that dataset field has extra fields that are not specified by COCO {extra_keys}.'
            '\nAfter saving the dataset unspecified fields will be lost.'
            '\nTo keep the unique fields extended one of existing objects or create new following one of the Protocols.'
        )


def isinstances(__obj: List[Any], __class_or_tuple: Union[Any, Tuple[Any]]) -> bool:
    """Return whether an every element of the list is an instance of a class or of a subclass thereof.
    A tuple, as in `isinstance(x, (A, B, ...))`, may be given as the target to check against.
    This is equivalent to `isinstance(x, A)` or `isinstance(x, B)` or ... etc.
    """
    return all(isinstance(obj, __class_or_tuple) for obj in __obj)