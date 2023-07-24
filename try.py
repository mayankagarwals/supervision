import supervision as sv

# import roboflow
# from roboflow import Roboflow
import json

# roboflow.login()
#
# rf = Roboflow()
#
# project = rf.workspace("athanasios-kokkinhs-iqro0").project("aedespupaeheads")
# dataset = project.version(1).download("coco")

ds = sv.DetectionDataset.from_coco(
    images_directory_path="/Users/mayankagarwal/Documents/Personal/CS/codebase/Supervision-Parent/coco-segmentation/train",
    annotations_path="/Users/mayankagarwal/Documents/Personal/CS/codebase/Supervision-Parent/coco-segmentation/train/_annotations.coco.json",
    force_masks=True
)

# image = ds.images['47_jpg.rf.bafe0a86cb20aa1fb74e406fe946ff5a.jpg']
# detections = ds.annotations['47_jpg.rf.bafe0a86cb20aa1fb74e406fe946ff5a.jpg']
#
#
# mask_annotator = sv.MaskAnnotator()
# annotated_image = mask_annotator.annotate(
#     scene=image.copy(),
#     detections=detections
# )

# sv.plot_image(image=annotated_image, size=(16, 16))


# original_json = json.load(open("../Supervision-Parent/coco-segmentation/train/_annotations.coco.json"))
# original_polygons = original_json["annotations"][0]["segmentation"]
# print(original_polygons)

ds.as_coco("../export_images","../export_coco/annotations.json")


def load_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# load the data from the files
data1 = load_json_file("../coco-segmentation/train/_annotations.coco.json")
data2 = load_json_file("../export_coco/annotations.json")

# check if the data from the two files is identical

if(len(data1["annotations"]) != len(data2["annotations"])):
    print("WRONG DATA FORMAT")

for i in range(len(data1["annotations"])):
    if(len( data1["annotations"][i]["segmentation"]) != len( data2["annotations"][i]["segmentation"])):
        print("WRONG DATA FORMAT")

print("Process completed")

# image = ds.images['47_jpg.rf.bafe0a86cb20aa1fb74e406fe946ff5a.jpg']
# detections = ds.annotations['47_jpg.rf.bafe0a86cb20aa1fb74e406fe946ff5a.jpg']
#
# mask_annotator = sv.MaskAnnotator()
# annotated_image = mask_annotator.annotate(
#     scene=image.copy(),
#     detections=detections
# )

# sv.plot_image(image=annotated_image, size=(16, 16))
# seems that sv can handle this issue

# exported_json = json.load(open("../export_coco/annotations.json"))
# exported_polygons = exported_json["annotations"][0]["segmentation"]
#
# print(len(original_polygons))
# print(len(exported_polygons))

'''

import numpy as np
import cv2
from typing import List, Optional

def mask_to_polygons(mask: np.ndarray) -> List[np.ndarray]:
    """
    Converts a binary mask to a list of polygons.

    Parameters:
        mask (np.ndarray): A binary mask represented as a 2D NumPy array of shape `(H, W)`,
            where H and W are the height and width of the mask, respectively.

    Returns:
        List[np.ndarray]: A list of polygons, where each polygon is represented by a NumPy array of shape `(N, 2)`,
            containing the `x`, `y` coordinates of the points. Polygons with fewer points than `MIN_POLYGON_POINT_COUNT = 3`
            are excluded from the output.
    """

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    return [
        np.squeeze(contour, axis=1)
        for contour in contours
        if contour.shape[0] >= 3
    ]




def filter_polygons_by_area(
    polygons: List[np.ndarray],
    min_area: Optional[float] = None,
    max_area: Optional[float] = None,
) -> List[np.ndarray]:
    """
    Filters a list of polygons based on their area.

    Parameters:
        polygons (List[np.ndarray]): A list of polygons, where each polygon is represented by a NumPy array of shape `(N, 2)`,
            containing the `x`, `y` coordinates of the points.
        min_area (Optional[float]): The minimum area threshold. Only polygons with an area greater than or equal to this value
            will be included in the output. If set to None, no minimum area constraint will be applied.
        max_area (Optional[float]): The maximum area threshold. Only polygons with an area less than or equal to this value
            will be included in the output. If set to None, no maximum area constraint will be applied.

    Returns:
        List[np.ndarray]: A new list of polygons containing only those with areas within the specified thresholds.
    """
    if min_area is None and max_area is None:
        return polygons
    ares = [cv2.contourArea(polygon) for polygon in polygons]
    return [
        polygon
        for polygon, area in zip(polygons, ares)
        if (min_area is None or area >= min_area)
        and (max_area is None or area <= max_area)
    ]



def approximate_polygon(
    polygon: np.ndarray, percentage: float, epsilon_step: float = 0.05
) -> np.ndarray:
    """
    Approximates a given polygon by reducing a certain percentage of points.

    This function uses the Ramer-Douglas-Peucker algorithm to simplify the input polygon by reducing the number of points
    while preserving the general shape.

    Parameters:
        polygon (np.ndarray): A 2D NumPy array of shape `(N, 2)` containing the `x`, `y` coordinates of the input polygon's points.
        percentage (float): The percentage of points to be removed from the input polygon, in the range `[0, 1)`.
        epsilon_step (float): Approximation accuracy step. Epsilon is the maximum distance between the original curve and its approximation.

    Returns:
        np.ndarray: A new 2D NumPy array of shape `(M, 2)`, where `M <= N * (1 - percentage)`, containing the `x`, `y` coordinates of the
            approximated polygon's points.
    """

    if percentage < 0 or percentage >= 1:
        raise ValueError("Percentage must be in the range [0, 1).")

    target_points = max(int(len(polygon) * (1 - percentage)), 3)

    if len(polygon) <= target_points:
        return polygon

    epsilon = 0
    approximated_points = polygon
    while True:
        epsilon += epsilon_step
        new_approximated_points = cv2.approxPolyDP(polygon, epsilon, closed=True)
        if len(new_approximated_points) > target_points:
            approximated_points = new_approximated_points
        else:
            break

    return np.squeeze(approximated_points, axis=1)



def approximate_mask_with_polygons(
    mask: np.ndarray,
    min_image_area_percentage: float = 0.0,
    max_image_area_percentage: float = 1.0,
    approximation_percentage: float = 0.75,
) -> List[np.ndarray]:
    height, width = mask.shape
    image_area = height * width
    minimum_detection_area = min_image_area_percentage * image_area
    maximum_detection_area = max_image_area_percentage * image_area

    polygons = mask_to_polygons(mask=mask)
    if len(polygons) == 1:
        polygons = filter_polygons_by_area(
            polygons=polygons, min_area=None, max_area=maximum_detection_area
        )
    else:
        polygons = filter_polygons_by_area(
            polygons=polygons,
            min_area=minimum_detection_area,
            max_area=maximum_detection_area,
        )
    return [
        approximate_polygon(polygon=polygon, percentage=approximation_percentage)
        for polygon in polygons
    ]



mask = np.ones((64,64))
a = approximate_mask_with_polygons(np.array(mask))
print(a)
'''