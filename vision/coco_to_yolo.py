import os 
import json
import yaml

def coco_to_yolo(input_path, output_path):
    with open(input_path + "annotations.json") as f:
        data = json.load(f)
    os.makedirs(output_path+ "images", exist_ok=True)
    os.makedirs(output_path+ "labels", exist_ok=True)
    images = {img["id"] for img in data["images"]}
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}
    annotations_by_image = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        annotations_by_image.setdefault(img_id, []).append(ann)
    for img_id, anns in annotations_by_image.item():
        img = images[img_id]
        w, h = img["width"], img["height"]
    label_path = os.path.join(output_path, "labels", img["file_name"].replace(".jpg", ".txt"))
    with open(label_path, "w") as f:
        for ann in anns:
            x,y,bw, bh =ann
            x_center= (x + bw/2) / w
            y_center= (y + bh/2) / h
            bw /= w
            bh /= h
            class_id = categories[ann["category_id"]]
            f.write(f"{class_id} {x_center} {y_center} {bw} {bh}\n")
    for filename in os.listdir(input_path + "images"):
        src = os.path.join(input_path, "images", filename)
        dst = os.path.join(output_path, "images", filename)
        os.symlink(src, dst)
if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    coco_to_yolo(config["input_path"], config["output_path"])