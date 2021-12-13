import json
import random

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, overload, cast

from structures import Image, Info, License
from structures.annotation import PanopticSegmentationAnnotation, dict_to_annotation
from structures.category import dict_to_category
from structures.types import (
    Annotation, Categorized, Category,
    Indexed, compare_dict_structure, isinstances
)


@dataclass
class COCO:
    """Dataclass that follows one of the COCO dataset structures.
    Has number of properties to easy work with COCO dataset.
    Such as appending new items, removing existing, reindexing, splitting or combining datasets, etc.
    """
    info: Info
    licenses: List[License] = field(default_factory=list)
    categories: Optional[List[Category]] = None
    images: List[Image] = field(default_factory=list)
    annotations: List[Annotation] = field(default_factory=list)


    @classmethod
    def from_dict(cls, coco_dict: Dict[str, Any], ignore_extra_keys = True):
        """Generates COCO datset dataclass from dictionary.

        Args:
            coco_dict (Dict[str, Any]): Dictionary objects that has COCO JSON dataset structure.
            ignore_extra_keys (bool, optional): Ignore the fact dictionary has more fields than specified in dataset. Defaults to True.

        Returns:
            COCO: COCO dataset generated from dictionary.
        """
        compare_dict_structure(coco_dict, cls)
        return cls(
            info = Info.from_dict(coco_dict['info'], ignore_extra_keys),
            licenses = [License.from_dict(license, ignore_extra_keys) for license in coco_dict['licenses']],
            images = [Image.from_dict(image, ignore_extra_keys) for image in coco_dict['images']],
            annotations = [dict_to_annotation(annotation) for annotation in coco_dict['annotations']],
            categories = (
                [dict_to_category(category, ignore_extra_keys) for category in coco_dict['categories']]
                if 'categories' in coco_dict.keys() 
                else None
            ),
        )

    @classmethod
    def from_json(cls, path_to_json: str, ignore_extra_keys = True):
        """Generates COCO datset dataclass from json file.

        Args:
            path_to_json (str): path where COCO JSON dataset is located.
            ignore_extra_keys (bool, optional): Ignore the fact dictionary has more fields than specified in dataset. Defaults to True.

        Returns:
            COCO: COCO dataset generated from JSON file.
        """
        with open(path_to_json) as f:
            data = json.load(f)
        return COCO.from_dict(data, ignore_extra_keys)


    def to_dict(self):
        """Transforms COCO dataclass to the Dictionary."""
        return {key: value for key, value in asdict(self) if value is not None}

    def to_json(self, path: str):
        """Saves current state of COCO dataset into json file, where `path` is absolute path with filename."""
        with open(path, 'w') as json_file:
            json.dump(self.to_dict(), json_file)



    @property
    def license_ids(self) -> List[int]:
        """List of all licenses ids."""
        return [license.id for license in self.licenses]

    @property
    def license_names(self) -> List[str]:
        """List of all licenses names."""
        return [license.name for license in self.licenses]

    @property
    def license_urls(self) -> List[str]:
        """List of all licenses urls."""
        return [license.url for license in self.licenses]


    @property
    def image_ids(self) -> List[int]:
        """List of all image ids."""
        return [image.id for image in self.images]

    @property
    def image_file_names(self) -> List[str]:
        """List of all image file names."""
        return [image.file_name for image in self.images]

    @property
    def image_flickr_urls(self) -> List[str]:
        """List of all image flickr urls."""
        return [image.flickr_url for image in self.images]

    @property
    def image_coco_urls(self) -> List[str]:
        """List of all image coco urls."""
        return [image.coco_url for image in self.images]


    @property
    def category_ids(self) -> List[int]:
        """List of all category ids if any present."""
        if self.categories:
            return [category.id for category in self.categories]
        return list()

    @property
    def category_names(self) -> List[str]:
        """List of all category names if any present."""
        if self.categories:
            return [category.name for category in self.categories]
        return list()

    @property
    def supercategories(self) -> List[str]:
        """List of all supercategories if any present."""
        if self.categories:
            return list({category.supercategory for category in self.categories})
        return list()


    @property
    def annotation_ids(self) -> List[int]:
        """List of all annotation ids if Annotations are Indexed."""
        return [
            annotation.id
            for annotation in self.annotations
            if isinstance(annotation, Indexed)
        ]



    def __post_init__(self):
        self._cached_license_ids: List[int] = list()
        self._cached_image_ids: List[int] = list()
        self._cached_category_ids: List[int] = list()
        self._cached_annotation_ids: List[int] = list()


    def __trim_cached_license_ids(self):
        self._cached_license_ids = [
            cached_id
            for cached_id in self._cached_license_ids
            if cached_id not in self.license_ids
        ]

    def __trim_cached_image_ids(self):
        self._cached_image_ids = [
            cached_id
            for cached_id in self._cached_image_ids
            if cached_id not in self.image_ids
        ]

    def __trim_cached_category_ids(self):
        self._cached_category_ids = [
            cached_id
            for cached_id in self._cached_category_ids
            if cached_id not in self.category_ids
        ]

    def __trim_cached_annotation_ids(self):
        self._cached_annotation_ids = [
            cached_id
            for cached_id in self._cached_annotation_ids
            if cached_id not in self.annotation_ids
        ]

    def __trim_all_cached_ids(self):
        self.__trim_cached_license_ids()
        self.__trim_cached_image_ids()
        self.__trim_cached_category_ids()
        self.__trim_cached_annotation_ids()


    def __generate_new_id(self, present_ids: List[int], cached_ids: List[int]) -> int:
        new_id = max(present_ids + cached_ids) + 1
        cached_ids.append(new_id)
        return new_id


    def __get_annotation_category_ids(self, annotation: Union[Categorized, PanopticSegmentationAnnotation]) -> List[int]:
        if isinstance(annotation, Categorized):
            return [annotation.category_id]

        elif isinstance(annotation, PanopticSegmentationAnnotation):
            return [segment_info.category_id for segment_info in annotation.segments_info]

        return [-1]


    def __get_annotations_category_ids(self, annotations: List[Union[Categorized, PanopticSegmentationAnnotation]]) -> List[int]:
        category_ids = []
        for annotation in annotations:
            category_ids += self.__get_annotation_category_ids(annotation)

        return category_ids


    def __combine_licenses(self, new_coco: 'COCO', ignore_duplicates = True):

        new_license_ids = []
        for license in new_coco.licenses:

            if not ignore_duplicates and (license.name in self.license_names):
                current_license = self.get_license(license.name)
                new_license_ids.append(current_license.id)
            else:
                new_license_ids.append(self.get_new_license_id())

        new_coco.reindex_licenses(new_license_ids)
        for license in new_coco.licenses:
            if not license.id in self.license_ids:
                self.append_license(license)


    def __combine_categories(self, new_coco: 'COCO', ignore_duplicates = True):
        if not new_coco.categories:
            return

        new_category_ids = []
        for category in new_coco.categories:

            if not ignore_duplicates and self.categories:
                if any([
                    category.name == self_category.name and category.supercategory == self_category.supercategory
                    for self_category in self.categories
                ]):
                    current_category = self.get_category(category.name)
                    if current_category:
                        new_category_ids.append(current_category.id)
                        continue

            new_category_ids.append(self.get_new_category_id())


        new_coco.reindex_categories(new_category_ids)
        for category in new_coco.categories:
            if not category.id in self.category_ids:
                self.append_category(category)


    def __combine_images(self, new_coco: 'COCO', ignore_duplicates = True):
        new_image_ids = []
        for image in new_coco.images:

            if not ignore_duplicates and (image.file_name in self.image_file_names):
                current_image = self.get_image(image.file_name)
                new_image_ids.append(current_image.id)
            else:
                new_image_ids.append(self.get_new_image_id())

        new_coco.reindex_images(new_image_ids)
        for image in new_coco.images:
            if not image.id in self.image_ids:
                self.append_image(image)



    @overload
    def get_license(self, id: int) -> License: ...
    @overload
    def get_license(self, name: str) -> License: ...
    @overload
    def get_license(self, image: Image) -> License: ...
    def get_license(self, input):
        """Gets License object based on its id, name or Image object.

        Args:
            input (int, str, Image): Id or name of License or Image object that is assigned to the license.

        Raises:
            TypeError: If input argument is not one of the specified types.

        Returns:
            License: Returns License based on specified input.
        """
        if isinstance(input, int):
            if input not in self.license_ids:
                return None

            return next(license for license in self.licenses if license.id == input)

        elif isinstance(input, str):
            if input not in self.license_names:
                return None

            return next(license for license in self.licenses if license.name == input)

        elif isinstance(input, Image):
            if input not in self.images:
                return None

            return next(license for license in self.licenses if license.id == input.license)

        raise TypeError('Input argument of unexpected type.')


    def get_licenses(self, images: List[Image]) -> List[License]:
        """Gets List of licenses based on List of image objects.

        Args:
            input (List[Image]): List of objects of type Image.

        Raises:
            TypeError: If input argument is not List of Images.

        Returns:
            List[License]: Returns List of licenses based on the given list of images.
        """
        if not isinstances(images, Image):
            raise TypeError('Input argument of unexpected type.')

        license_ids = {self.get_license(image).id for image in images}
        return [self.get_license(id) for id in license_ids]


    @overload
    def get_category(self, id: int) -> Optional[Category]: ...
    @overload
    def get_category(self, name: str) -> Optional[Category]: ...
    @overload
    def get_category(self, annotation: Categorized) -> Optional[Category]: ...
    @overload
    def get_category(self, annotation: PanopticSegmentationAnnotation) -> Optional[Category]: ...
    def get_category(self, input):
        """Gets Category like object based on its id, name or Annotaiton.

        Args:
            input (int, str, Categorized, PanopticSegmentationAnnotation):
                Id, name of Category or Cagetorized Annotation.

        Raises:
            TypeError: If input argument is not one of the specified types.

        Returns:
            License: Returns Category based on specified input.
        """
        if not self.categories:
            return None

        if isinstance(input, int):
            if input not in self.category_ids:
                return None

            return next(category for category in self.categories if category.id == input)

        elif isinstance(input, str):
            if input not in self.category_names:
                return None

            return next(category for category in self.categories if category.name == input)

        elif isinstance(input, (Categorized, PanopticSegmentationAnnotation)):
            if input not in self.annotations:
                return None

            return next(category for category in self.categories if category.id in self.__get_annotation_category_ids(input))

        raise TypeError('Input argument of unexpected type.')


    @overload
    def get_categories(self, supercategory: str) -> List[Category]: ...
    @overload
    def get_categories(self, image: Image) -> List[Category]: ...
    @overload
    def get_categories(self, images: List[Image]) -> List[Category]: ...
    @overload
    def get_categories(self, annotations: List[Categorized]) -> List[Category]: ...
    @overload
    def get_categories(self, annotations: List[PanopticSegmentationAnnotation]) -> List[Category]: ...
    def get_categories(self, input):
        """Gets List of categories based on either of: supercategory, image, list of images, list of annotations.

        Args:
            input (str, Image, List[Image], List[Categorized], List[PanopticSegmentationAnnotation]): input to be used to get the categories.
                Can be supercategory, or image, or List of images, or List of annotations.

        Raises:
            TypeError: If input argument is not one of the specified types.

        Returns:
            List[Category]: Returns List of categories based on the given input.
        """
        if not self.categories:
            return list()

        if isinstance(input, str):
            return [category for category in self.categories if category.supercategory == input]

        elif isinstance(input, Image):
            return [
                category
                for category in self.categories
                if category.id in self.__get_annotations_category_ids(self.get_annotations(input))
            ]

        elif isinstance(input, list):
            if isinstances(input, Image):
                return [
                    category
                    for category in self.categories
                    if category.id in self.__get_annotations_category_ids(self.get_annotations(input))
                ]

            elif isinstances(input, (Categorized, PanopticSegmentationAnnotation)):
                return [
                    category
                    for category in self.categories
                    if category.id in self.__get_annotations_category_ids(input)
                ]

        raise TypeError('Input argument of unexpected type.')


    @overload
    def get_image(self, id: int) -> Image: ...
    @overload
    def get_image(self, file_name: str) -> Image: ...
    @overload
    def get_image(self, annotation: Annotation) -> Image: ...
    def get_image(self, input):
        """Gets Image object based on its id, file name or Annotaiton.

        Args:
            input (int, str, Annotation):
                Id, file name of Image or Annotation.

        Raises:
            TypeError: If input argument is not one of the specified types.

        Returns:
            Image: Returns image based on specified input.
        """
        if isinstance(input, int):
            if input not in self.image_ids:
                return None

            return next(image for image in self.images if image.id == input)

        elif isinstance(input, str):
            if input not in self.image_file_names:
                return None

            return next(image for image in self.images if image.file_name == input)

        elif isinstance(input, Annotation):
            if input not in self.annotations:
                return None

            return next(image for image in self.images if image.id == input.image_id)

        raise TypeError('Input argument of unexpected type.')


    @overload
    def get_images(self, license_id: int) -> List[Image]: ...
    @overload
    def get_images(self, license_name: str) -> List[Image]: ...
    @overload
    def get_images(self, license: License) -> List[Image]: ...
    @overload
    def get_images(self, category: Category) -> List[Image]: ...
    @overload
    def get_images(self, annotations: List[Annotation]) -> List[Image]: ...
    @overload
    def get_images(self, categories: List[Category]) -> List[Image]: ...
    def get_images(self, input):
        """Gets List of images based on either of: license id, license name, License, Category, List of Annotations, List of Categoires.

        Args:
            input (int, str, License, Category, List[Annotation], List[Category]): input to be used to get the categories.
                Can be License information, category like object or list of objects, or list of annotations.

        Raises:
            TypeError: If input argument is not one of the specified types.

        Returns:
            List[Image]: Returns List of images based on the given input.
        """
        if isinstance(input, int):
            return [image for image in self.images if image.license == input]

        elif isinstance(input, str):
            return [image for image in self.images if image.license == self.get_license(input).id]

        elif isinstance(input, License):
            return [image for image in self.images if image.license == input.id]

        elif isinstance(input, Category) and self.categories:
            return [
                image
                for image in self.images
                if input.id in [category.id for category in self.get_categories(image)]
            ]

        elif isinstance(input, list):
            if isinstances(input, Annotation):
                return [
                    image
                    for image in self.images
                    if image.id in [annotation.image_id for annotation in input]
                ]

            elif isinstances(input, Category) and self.categories:
                return sum([self.get_images(category) for category in input], [])

        raise TypeError('Input argument of unexpected type.')


    def get_annotation(self, id: int) -> Annotation:
        """Gets Annotation like object based on its id, file name or Annotaiton.

        Args:
            input (int): Id of Annotation.

        Raises:
            TypeError: If input argument is not int.

        Returns:
            Annotation: Returns annotation based on specified id.
        """
        if id in self.annotation_ids and isinstance(id, int):
            return next(
                cast(Annotation, annotation)
                for annotation in self.annotations
                if isinstance(annotation, Indexed)
                if annotation.id == id
            )

        raise TypeError('Input argument of unexpected type.')

    @overload
    def get_annotations(self, image_id: int) -> List[Annotation]: ...
    @overload
    def get_annotations(self, file_name: str) -> List[Annotation]: ...
    @overload
    def get_annotations(self, image: Image) -> List[Annotation]: ...
    @overload
    def get_annotations(self, images: List[Image]) -> List[Annotation]: ...
    def get_annotations(self, input):
        """Gets List of annotations based on Image information either of: image id, file name, image, list of images.

        Args:
            input (int, str, Image, List[Image]): input to be used to get the annotations.
                Can be Image information, such as id, file name, or objects itself.

        Raises:
            TypeError: If input argument is not one of the specified types.

        Returns:
            List[Annotation]: Returns List of annotations based on the given input.
        """
        if isinstance(input, int):
            return [annotation for annotation in self.annotations if annotation.image_id == input]

        elif isinstance(input, str):
            return [
                annotation
                for annotation in self.annotations
                if self.get_image(annotation.image_id).file_name == input
            ]

        elif isinstance(input, Image):
            return [annotation for annotation in self.annotations if annotation.image_id == input.id]

        elif isinstance(input, list):
            if isinstances(input, Image):
                return [
                    annotation
                    for annotation in self.annotations
                    if annotation.image_id in [image.id for image in input]
                ]

        raise TypeError('Input argument of unexpected type.')

    @overload
    def get_annotations_by_category(self, category_id: int) -> List[Annotation]: ...
    @overload
    def get_annotations_by_category(self, category_name: str) -> List[Annotation]: ...
    @overload
    def get_annotations_by_category(self, category: Category) -> List[Annotation]: ...
    @overload
    def get_annotations_by_category(self, categories: List[Category]) -> List[Annotation]: ...
    def get_annotations_by_category(self, input):
        """Gets List of annotations based on Category information either of: category id, category name, category, list of categories.

        Args:
            input (int, str, Category, List[Category]): input to be used to get the annotations.
                Can be Category information, such as id, file name, or objects itself.

        Raises:
            TypeError: If input argument is not one of the specified types.

        Returns:
            List[Annotation]: Returns List of annotations based on the given input.
        """
        if not self.categories or not isinstances(self.annotations, (Categorized, PanopticSegmentationAnnotation)):
            return list()

        if isinstance(input, int):
            return [annotation for annotation in self.annotations if annotation.category_id == input]

        elif isinstance(input, str):
            return [
                annotation
                for annotation in self.annotations
                if self.get_category(annotation.category_id).name == input
            ]

        elif isinstance(input, Category):
            return [
                annotation for annotation in self.annotations 
                if input.id in self.__get_annotation_category_ids(annotation)
            ]

        elif isinstance(input, list):
            if isinstances(input, Category):
                return [
                    annotation
                    for annotation in self.annotations
                    if set([category.id for category in input]).intersection(
                        self.__get_annotation_category_ids(annotation)
                    )
                ]

        raise TypeError('Input argument of unexpected type.')

    def get_annotations_by_supercategory(self, supercategory: str) -> List[Annotation]:
        """Gets List of annotations based on supercategory.

        Args:
            input (str): supercategory name.

        Raises:
            TypeError: If input argument is not str.

        Returns:
            List[Annotation]: Returns List of annotations based on supercategory.
        """
        if not isinstance(supercategory, str):
            raise TypeError('Input argument of unexpected type.')

        if not self.categories or not isinstances(self.annotations, (Categorized, PanopticSegmentationAnnotation)):
            return list()

        return [
            annotation
            for annotation in self.annotations
            if isinstance(annotation, (Categorized, PanopticSegmentationAnnotation))
            if set(self.__get_annotation_category_ids(annotation)).intersection([
                category.id for category in self.get_categories(supercategory)
            ])
        ]


    def clear_cached_ids(self):
        """Clears cached ids that were generated by the `get_new_[]_id` methods."""
        self._cached_license_ids: List[int] = list()
        self._cached_image_ids: List[int] = list()
        self._cached_category_ids: List[int] = list()
        self._cached_annotation_ids: List[int] = list()

    def get_new_license_id(self) -> int:
        """Generate new id for the license."""
        return self.__generate_new_id(self.license_ids, self._cached_license_ids)

    def get_new_category_id(self) -> int:
        """Generate new id for the category."""
        return self.__generate_new_id(self.category_ids, self._cached_category_ids)

    def get_new_image_id(self) -> int:
        """Generate new id for the image."""
        return self.__generate_new_id(self.image_ids, self._cached_image_ids)

    def get_new_annotation_id(self) -> int:
        """Generate new id for the annotation."""
        return self.__generate_new_id(self.annotation_ids, self._cached_annotation_ids)



    def append_license(self, license: License):
        """Appends License object to the dataset. Consider reindexing license with the `get_new_license_id()`.

        Args:
            license (License): New License that will be appended to the dataset.

        Raises:
            TypeError: If passed argument is not of type License
            ValueError: If License with given Id already present.
        """
        if not isinstance(license, License):
            raise TypeError('Input argument of unexpected type.')

        if license.id in self.license_ids:
            raise ValueError(
                f"License with ID {license.id} already exists."
                "Consider using get_new_license_id() function when assigning ID to the object."
            )

        self.licenses.append(license)
        self.__trim_cached_license_ids()

    def extend_licenses(self, licenses: List[License]):
        """Extends dataset with the given Licenses. Consider reindexing licenses with the `get_new_license_id()`.

        Args:
            licenses (List[License]): New Licenses that will extend current dataset.

        Raises:
            TypeError: If passed arguments is not of type License
            ValueError: If one of the Licenses with given Id already present.
        """
        [self.append_license(license) for license in licenses]


    def append_category(self, category: Category):
        """Appends Category like object to the dataset. Consider reindexing category with the `get_new_category_id()`.

        Args:
            category (Category): New Category that will be appended to the dataset.

        Raises:
            TypeError: If passed argument is not of type Category
            ValueError: If Category with given Id already present.
        """
        if not self.categories:
            self.categories = list()

        if not isinstance(category, Category):
            raise TypeError('Input argument of unexpected type.')

        if category.id in self.category_ids:
            raise ValueError(
                f"Category with ID {category.id} already exists."
                "Consider using get_new_category_id() function when assigning ID to the object."
            )

        self.categories.append(category)
        self.__trim_cached_category_ids()

    def extend_categories(self, categories: List[Category]):
        """Extends dataset with the given Categories. Consider reindexing categories with the `get_new_category_id()`.

        Args:
            categories (List[Category]): New Categories that will extend current dataset.

        Raises:
            TypeError: If passed arguments is not of type Category
            ValueError: If one of the Categories with given Id already present.
        """
        [self.append_category(category) for category in categories]


    def append_image(self, image: Image):
        """Appends Image object to the dataset. Consider reindexing image with the `get_new_image_id()`.

        Args:
            image (Category): New Category that will be appended to the dataset.

        Raises:
            TypeError: If passed argument is not of type Image
            ValueError: If Image with given Id already present.
        """
        if not isinstance(image, Image):
            raise TypeError('Input argument of unexpected type.')

        if image.id in self.image_ids:
            raise ValueError(
                f"Image with ID {image.id} already exists."
                "Consider using get_new_image_id() function when assigning ID to the object."
            )

        self.images.append(image)
        self.__trim_cached_image_ids()

    def extend_images(self, images: List[Image]):
        """Extends dataset with the given Images. Consider reindexing images with the `get_new_image_id()`.

        Args:
            images (List[Image]): New Images that will extend current dataset.

        Raises:
            TypeError: If passed arguments is not of type Image
            ValueError: If one of the Images with given Id already present.
        """
        [self.append_image(image) for image in images]


    def append_annotation(self, annotation: Annotation):
        """Appends Annotation like object to the dataset. Consider reindexing annotation with the `get_new_annotation_id()`.

        Args:
            annotation (Annotation): New Annotation that will be appended to the dataset.

        Raises:
            TypeError: If passed argument is not of type Annotation
            ValueError: If Annotation with given Id already present.
        """
        if not isinstance(annotation, Annotation):
            raise TypeError('Input argument of unexpected type.')

        if isinstance(annotation, Indexed):
            if annotation.id in self.annotation_ids:
                raise ValueError(
                    f"Annotation with ID {annotation.id} already exists."
                    "Consider using get_new_annotation_id() function when assigning ID to the object."
                )

        self.annotations.append(annotation)
        self.__trim_cached_annotation_ids()

    def extend_annotations(self, annotations: List[Annotation]):
        """Extends dataset with the given Annotations. Consider reindexing annotations with the `get_new_annotation_id()`.

        Args:
            annotations (List[Annotation]): New Annotations that will extend current dataset.

        Raises:
            TypeError: If passed arguments is not of type Annotation
            ValueError: If one of the Annotations with given Id already present.
        """
        [self.append_annotation(annotation) for annotation in annotations]



    def remove_license(self, license: License, remove_images = True, remove_annotations = True):
        """Removes License from the dataset and corresponding connections.

        Args:
            license (License): License to remove.
            remove_images (bool, optional): Remove images that corresponds to this license. Defaults to True.
            remove_annotations (bool, optional): Remove annotations that corresponds to removed images. Defaults to True.
        """
        if license not in self.licenses:
            return

        self.licenses.remove(license)
        if remove_images:
            self.remove_images(self.get_images(license), remove_annotations)

    def remove_licenses(self, licenses: List[License], remove_images = True, remove_annotations = True):
        """Removes Licenses from the dataset and corresponding connections.

        Args:
            licenses (List[License]): Licenses to remove.
            remove_images (bool, optional): Remove images that corresponds to the licenses. Defaults to True.
            remove_annotations (bool, optional): Remove annotations that corresponds to removed images. Defaults to True.
        """
        [self.remove_license(license, remove_images, remove_annotations) for license in licenses]

    def remove_category(self, category: Category, remove_annotations = True):
        """Removes Category from the dataset and corresponding connections.

        Args:
            category (Category): Category to remove.
            remove_annotations (bool, optional): Remove annotations that corresponds to removed category. Defaults to True.
        """
        if self.categories:
            if category not in self.categories:
                return

            self.categories.remove(category)

            if remove_annotations:
                self.remove_annotaitons(self.get_annotations_by_category(category))

    def remove_categories(self, categories: List[Category], remove_annotations = True):
        """Removes Categories from the dataset and corresponding connections.

        Args:
            categories (List[Category]): Categories to remove.
            remove_annotations (bool, optional): Remove annotations that corresponds to removed categories. Defaults to True.
        """
        [self.remove_category(category, remove_annotations) for category in categories]

    def remove_image(self, image: Image, remove_annotations = True):
        """Removes Image from the dataset and corresponding connections.

        Args:
            image (Image): Image to remove.
            remove_annotations (bool, optional): Remove annotations that corresponds to removed image. Defaults to True.
        """
        if image not in self.images:
            return

        self.images.remove(image)
        if remove_annotations:
            self.remove_annotaitons(self.get_annotations(image))

    def remove_images(self, images: List[Image], remove_annotations = True):
        """Removes Images from the dataset and corresponding connections.

        Args:
            images (List[Image]): Images to remove.
            remove_annotations (bool, optional): Remove annotations that corresponds to removed images. Defaults to True.
        """
        [self.remove_image(image, remove_annotations) for image in images]

    def remove_annotation(self, annotation: Annotation):
        """Removes Annotation from the dataset.

        Args:
            annotation (Annotation): Annotation to remove.
        """
        if annotation not in self.annotations:
            return

        self.annotations.remove(annotation)

    def remove_annotaitons(self, annotations: List[Annotation]):
        """Removes Annotations from the dataset.

        Args:
            annotations (List[Annotation]): Annotations to remove.
        """
        [self.remove_annotation(annotation) for annotation in annotations]


    def clean_dataset(self):
        """Cleans dataset from unused elements, and reindex all elements and their connections."""
        self.trim_dataset()
        self.reindex_dataset()

    def trim_dataset(self):
        """Trims dataset by removing unused annotations, images, categories, licenses."""
        self.trim_annotations()
        self.trim_images()
        self.trim_categories()
        self.trim_licenses()

    def trim_licenses(self):
        """Removes unused licenses."""
        self.licenses = [
            license
            for license in self.licenses
            if license.id in [image.license for image in self.images]
        ]

    def trim_categories(self):
        """Removes unused categories."""
        if self.categories and isinstances(self.annotations, Categorized):
            self.categories = [
                category
                for category in self.categories
                if category.id in [annotation.category_id for annotation in self.annotations]
            ]

    def trim_images(self):
        """Removes unused images."""
        self.images = [
            image
            for image in self.images
            if image.id in [annotation.image_id for annotation in self.annotations]
            and image.license in self.license_ids
        ]

    def trim_annotations(self):
        """Removes unused annotations."""
        self.annotations = [
            annotation
            for annotation in self.annotations
            if annotation.image_id in self.image_ids
        ]



    def reindex_dataset(self):
        """Reindexes entire dataset and removes cached ids."""
        self.reindex_annotations()
        self.reindex_images()
        self.reindex_categories()
        self.reindex_licenses()
        self.clear_cached_ids()


    def reindex_license(self, license: License, new_id: int = None):
        """Reindexes passed mutable license with the given new_id, or generates new one.

        Args:
            license (License): License that will be reindexed.
            new_id (int, optional): New id to be used if None, new will be generated. Defaults to None.
        """
        if new_id is None:
            new_id = self.get_new_license_id()

        for image in self.get_images(license):
            image.license = new_id

        license.id = new_id

    def reindex_licenses(self, new_ids: List[int] = None):
        """Reindexes licenses with the given list of ids. new_ids should have the same length as there are licenses.
        If no new_ids passed, new incremented one will be added.

        Args:
            new_ids (List[int], optional): List of ids that will be used for reindexing if None range will be generated. Defaults to None.

        Raises:
            ValueError: If length of ids is different from length of licenses.
        """
        if not new_ids:
            new_ids = list(range(len(self.licenses)))

        if len(new_ids) != len(self.licenses):
            raise ValueError('Inapropirate amount of ids. Length of new_ids should be equal to the length of licenses.')

        for new_id in new_ids:
            if new_id in self.license_ids:
                duplicated_license = self.get_license(new_id)
                self.reindex_license(duplicated_license, self.get_new_license_id())

        for new_id, license in zip(new_ids, self.licenses):
            self.reindex_license(license, new_id)


    def reindex_category(self, category: Category, new_id: int = None):
        """Reindexes passed mutable category with the given new_id, or generates new one.

        Args:
            category (Category): Category that will be reindexed.
            new_id (int, optional): New id to be used if None, new will be generated. Defaults to None.
        """
        if not self.categories:
            return

        if new_id is None:
            new_id = self.get_new_category_id()

        for annotation in self.get_annotations_by_category(category):
            if isinstance(annotation, Categorized):
                annotation.category_id = new_id

            elif isinstance(annotation, PanopticSegmentationAnnotation):
                for segment_info in annotation.segments_info:
                    if segment_info.category_id == category.id:
                        segment_info.category_id = new_id

        category.id = new_id

    def reindex_categories(self, new_ids: List[int] = None):
        """Reindexes categories with the given list of ids. new_ids should have the same length as there are categories.
        If no new_ids passed, new incremented one will be added.

        Args:
            new_ids (List[int], optional): List of ids that will be used for reindexing if None range will be generated. Defaults to None.

        Raises:
            ValueError: If length of ids is different from length of categories.
        """
        if not self.categories:
            return

        if not new_ids:
            new_ids = list(range(len(self.categories)))

        if len(new_ids) != len(self.categories):
            raise ValueError('Inapropirate amount of ids. Length of new_ids should be equal to the length of categories.')

        for new_id, category in zip(new_ids, self.categories):
            if new_id in self.category_ids:
                duplicated_category = self.get_category(new_id)
                if duplicated_category:
                    self.reindex_category(duplicated_category, self.get_new_category_id())

            self.reindex_category(category, new_id)


    def reindex_image(self, image: Image, new_id: int):
        """Reindexes passed mutable image with the given new_id, or generates new one.

        Args:
            image (Image): Image that will be reindexed.
            new_id (int, optional): New id to be used if None, new will be generated. Defaults to None.
        """
        if new_id is None:
            new_id = self.get_new_image_id()

        for annotation in self.get_annotations(image):
            annotation.image_id = new_id

        image.id = new_id

    def reindex_images(self, new_ids: List[int] = None):
        """Reindexes images with the given list of ids. new_ids should have the same length as there are images.
        If no new_ids passed, new incremented one will be added.

        Args:
            new_ids (List[int], optional): List of ids that will be used for reindexing if None range will be generated. Defaults to None.

        Raises:
            ValueError: If length of ids is different from length of images.
        """
        if not new_ids:
            new_ids = list(range(len(self.images)))

        if len(new_ids) != len(self.images):
            raise ValueError('Inapropirate amount of ids. Length of new_ids should be equal to the length of images.')

        for new_id, image in zip(new_ids, self.images):

            if new_id in self.image_ids:
                duplicated_image = self.get_image(new_id)
                self.reindex_image(duplicated_image, self.get_new_image_id())

            self.reindex_image(image, new_id)


    def reindex_annotation(self, annotation: Annotation, new_id: int):
        """Reindexes passed mutable annotation with the given new_id, or generates new one.

        Args:
            annotation (Annotation): Annotation that will be reindexed.
            new_id (int, optional): New id to be used if None, new will be generated. Defaults to None.
        """
        if new_id is None:
            new_id = self.get_new_annotation_id()

        if isinstance(annotation, Indexed):
            annotation.id = new_id

    def reindex_annotations(self, new_ids: List[int] = None):
        """Reindexes annotations with the given list of ids. new_ids should have the same length as there are annotations.
        If no new_ids passed, new incremented one will be added.

        Args:
            new_ids (List[int], optional): List of ids that will be used for reindexing if None range will be generated. Defaults to None.

        Raises:
            ValueError: If length of ids is different from length of annotations.
        """
        if not new_ids:
            new_ids = list(range(len(self.annotations)))

        if len(new_ids) != len(self.annotations):
            raise ValueError('Inapropirate amount of ids. Length of new_ids should be equal to the length of annotations.')

        for new_id, annotation in zip(new_ids, self.annotations):

            if new_id in self.annotation_ids:
                duplicated_annotation = self.get_annotation(new_id)
                self.reindex_annotation(duplicated_annotation, self.get_new_annotation_id())

            self.reindex_annotation(annotation, new_id)



    def combine(self, new_coco: 'COCO', ignore_duplicates = True):
        """Combines another COCO dataset with the current one.
        Info stays the same, however Licenses, categories, images and annotations are combined.

        Args:
            new_coco (COCO): Another COCO dataset that will be combined with current one.
            ignore_duplicates (bool, optional): If True duplicated licenses, categories, images and annotations will be kept.
                If False duplicated elements will be removed. Defaults to True.
        """
        new_coco = deepcopy(new_coco)

        self.__combine_licenses(new_coco, ignore_duplicates)
        self.__combine_categories(new_coco, ignore_duplicates)
        self.__combine_images(new_coco, ignore_duplicates)

        new_coco.reindex_annotations([self.get_new_annotation_id() for _ in new_coco.annotations])
        self.extend_annotations(new_coco.annotations)

        self.clear_cached_ids()


    def split(self, percentage: float, seed: int = 101) -> Tuple['COCO']:
        """Splits dataset in two parts based on the specified percentage.
        Splitting process is perfromed on images.

        Args:
            percentage (float): Value between 0 and 1 that that will be used during splitting.
            seed (int): random seed.

        Raises:
            ValueError: If percentage arguments is not between 0 and 1

        Returns:
            Tuple[COCO]: Tuple of new two COCO datasets generated from the current one based on the percentage.
        """
        if percentage > 1.0 or percentage < 0.0:
            raise ValueError('percentage should be between 0 and 1.')

        images = deepcopy(self.images)

        random.seed(seed)
        random.shuffle(images)

        point = int(len(self.images) * percentage)
        datasets: Any = tuple()

        for subimages in [images[:point], images[point:]]:
            datasets += COCO(
                info = deepcopy(self.info),
                licenses = deepcopy(self.get_licenses(subimages)),
                images = subimages,
                annotations = deepcopy(self.get_annotations(subimages)),
                categories = deepcopy(self.get_categories(subimages))
            ),

        return datasets


    def category_based_split(self, percentage: float, seed: int = 101) -> Tuple['COCO']:
        """Not Implemented!"""
        raise NotImplementedError()
