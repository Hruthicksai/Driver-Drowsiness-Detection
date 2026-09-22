import os
import cv2
import numpy as np

from config import (
    YOLO_CONFIDENCE,
    YOLO_NMS_THRESHOLD,
    PHONE_CLASS_NAME
)


class PhoneDetector:

    def __init__(
        self,
        config_path,
        weights_path,
        names_path
    ):
        self.enabled = False

        if not (os.path.exists(config_path) and os.path.exists(weights_path) and os.path.exists(names_path)):
            print("[INFO] YOLO files not found. Phone detection is disabled.")
            print("       Run 'python download_models.py' to enable phone detection.")
            return

        try:
            self.net = cv2.dnn.readNet(
                weights_path,
                config_path
            )

            with open(names_path, "r") as file:
                self.classes = [
                    line.strip()
                    for line in file.readlines()
                ]

            try:
                self.output_layers = self.net.getUnconnectedOutLayersNames()
            except Exception:
                layer_names = self.net.getLayerNames()
                output_layers = self.net.getUnconnectedOutLayers()
                self.output_layers = [
                    layer_names[int(i) - 1 if isinstance(i, (int, np.integer)) else int(i[0]) - 1]
                    for i in output_layers
                ]
            self.enabled = True
        except Exception as e:
            print(f"[WARNING] Could not load YOLO model: {e}. Phone detection disabled.")
            self.enabled = False


    def detect(self, frame):
        if not getattr(self, "enabled", False):
            return False

        height, width = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame,
            1 / 255.0,
            (416, 416),
            swapRB=True,
            crop=False
        )

        self.net.setInput(blob)

        outputs = self.net.forward(
            self.output_layers
        )

        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:

            for detection in output:

                scores = detection[5:]

                class_id = int(
                    scores.argmax()
                )

                confidence = scores[class_id]

                if confidence > YOLO_CONFIDENCE:

                    center_x = int(
                        detection[0] * width
                    )

                    center_y = int(
                        detection[1] * height
                    )

                    w = int(
                        detection[2] * width
                    )

                    h = int(
                        detection[3] * height
                    )

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            YOLO_CONFIDENCE,
            YOLO_NMS_THRESHOLD
        )

        phone_detected = False

        if len(indexes) > 0:

            for i in indexes.flatten():

                label = self.classes[class_ids[i]]

                if label == PHONE_CLASS_NAME:

                    phone_detected = True

                    x, y, w, h = boxes[i]

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 0, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"PHONE {confidences[i]:.2f}",
                        (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

        return phone_detected