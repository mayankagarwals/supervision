import supervision as sv
import json

ds = sv.DetectionDataset.from_coco(
    images_directory_path="/Users/mayankagarwal/Documents/Personal/CS/codebase/Supervision-Parent/coco-segmentation/train",
    annotations_path="/Users/mayankagarwal/Documents/Personal/CS/codebase/Supervision-Parent/coco-segmentation/train/_annotations.coco.json",
    force_masks=True
)
