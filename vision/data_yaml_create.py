import yaml

data = {
    "path": "path/to/dataset",
    "train": "train/images",
    "val": "val/images",
    "test": "test/images",
    "nc": 80,
    "names": ["class1", "class2", "class3", "class4", "class5", "class6", "class7", "class8", "class9", "class10",
              "class11", "class12", "class13", "class14", "class15", "class16", "class17", "class18", "class19", "class20",
              "class21", "class22", "class23", "class24", "class25", "class26", "class27", "class28", "class29", "class30",
              "class31", "class32", "class33", "class34", "class35", "class36", "class37", "class38", "class39", "class40",
              "class41", "class42", "class43", "class44", "class45", "class46", "class47", "class48", "class49",
              "class50","class51","class52","class53","class54","class55","class56","class57","class58","class59",
              "class60","class61","class62","class63","class64","class65","class66","class67","class68","class69",
              "class70","class71","class72","class73"," class74"," class75"," class76"," class77"," class78",
              " class79"]
}
with open("data.yaml", "w") as f:
    yaml.dump(data, f)