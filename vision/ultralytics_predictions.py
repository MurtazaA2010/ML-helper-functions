def generate_predictions(results):
    rows = []
    for r in results:
        img_name = os.path.basename(os.path.splitext(r.path)[0].strip())
        if r.boxes is None or len(r.boxes) == 0:
            continue
        boxes = r.boxes.xywhn.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy()
        idx = confs.argmax()
        box = boxes[idx]
        conf = confs[idx] ## for single class prediction, we take the one with the highest confidence
        ## for multi class predction do the following:
        ## preds = []
        ## t = 0.5 # confidence threshold
        ## for box, conf, cls in zip(boxes, confs, cls):
        ##     if conf > t:
        ##         x1, y1, w,h = box
        ##         class_id = int(cls)
        ##         pred = f"{conf:.4f} {int(c)} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}"
        ##          preds.append(pred)
        ## rows.append({
        ##     "img_name": img_name,
        ##     "predictions": preds})
        ## PredictionString = ";".join(preds)
        
        class_id = int(cls[idx])
        x1, y1, w,h, = box
        rows.append({
            "img_name": img_name,
            "class_id": class_id,
            "confidence": conf,
            "x1": x1,
            "y1": y1,
            "w": w,
            "h": h
        })
        return rows

if __name__ == "__main__":
    import os
    from ultralytics import YOLO
    model = YOLO("yolov8m.pt")
    results = model("path/to/images")
    predictions = generate_predictions(results)