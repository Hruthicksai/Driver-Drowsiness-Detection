import time
import winsound
import cv2

from config import ALERT_COOLDOWN


class AlertSystem:

    def __init__(self):
        self.last_alert_time = 0

    def alert(self, alert_type):

        current_time = time.time()

        if current_time - self.last_alert_time < ALERT_COOLDOWN:
            return

        self.last_alert_time = current_time

        print(f"ALERT: {alert_type}")

        try:
            winsound.Beep(1000, 700)
            winsound.Beep(1200, 700)

        except Exception:
            print("\a")


def visual_alert(frame, message):

    height, width = frame.shape[:2]

    cv2.rectangle(
        frame,
        (0, 0),
        (width, 75),
        (0, 0, 255),
        -1
    )

    cv2.putText(
        frame,
        message,
        (25, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        3
    )