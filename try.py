import supervision as sv
import json

ds = sv.DetectionDataset.from_coco(
    images_directory_path="/Users/mayankagarwal/Documents/Personal/CS/codebase/Supervision-Parent/coco-segmentation/train",
    annotations_path="/Users/mayankagarwal/Documents/Personal/CS/codebase/Supervision-Parent/coco-segmentation/train/_annotations.coco.json",
    force_masks=True
)


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
    raise Exception("WRONG DATA FORMAT")

for i in range(len(data1["annotations"])):
    if(len( data1["annotations"][i]["segmentation"]) != len( data2["annotations"][i]["segmentation"])):
        raise Exception("WRONG DATA FORMAT")

print("Process completed")


