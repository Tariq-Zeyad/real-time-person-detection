import torch
import numpy as np
import cv2
import time

from ultralytics import YOLO
import supervision as sv

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from email_settings import passwords, from_email, to_email


# ---------------- EMAIL ----------------
def check_smtp_connection():
    try:
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(from_email, passwords)
        return server
    except Exception as e:
        print(f"SMTP Error: {e}")
        return None


def send_email(server, people_count):
    message = MIMEMultipart()
    message["From"] = from_email
    message["To"] = to_email
    message["Subject"] = "Security Alert"

    message.attach(
        MIMEText(f"Alert: {people_count} person(s) detected!")
    )

    server.sendmail(from_email, to_email, message.as_string())


# ---------------- DETECTION CLASS ----------------
class ObjectDetection:
    def __init__(self, capture_index=0):

        self.capture_index = capture_index

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.model = YOLO("yolov8n.pt")
        self.model.fuse()

        self.box_annotator = sv.BoxAnnotator(thickness=2)

        self.email_sent = False
        self.server = check_smtp_connection()

        if self.server:
            print("SMTP Connected Successfully")
        else:
            print("SMTP Connection Failed")

    # --------- PREDICTION ----------
    def predict(self, frame):
        return self.model(frame)

    # --------- DRAW BOXES ----------
    def plot_boxes(self, result, frame):

        detections = sv.Detections.from_ultralytics(result)

        person_detections = detections[detections.class_id == 0]

        frame = self.box_annotator.annotate(
            scene=frame,
            detections=person_detections
        )

        return frame, person_detections

    # --------- MAIN LOOP ----------
    def __call__(self):

        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:

            start_time = time.time()

            ret, frame = cap.read()
            assert ret

            results = self.predict(frame)

            frame, detections = self.plot_boxes(results[0], frame)

            # -------- EMAIL ALERT --------
            if len(detections) > 0:
                if not self.email_sent and self.server:
                    send_email(self.server, len(detections))
                    self.email_sent = True
            else:
                self.email_sent = False

            # -------- FPS --------
            end_time = time.time()
            fps = 1 / np.round(end_time - start_time, 2)

            cv2.putText(
                frame,
                f"FPS: {int(fps)}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("YOLOv8 Person Detection", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

        if self.server:
            self.server.quit()


# ---------------- RUN ----------------
if __name__ == "__main__":
    detector = ObjectDetection(capture_index=0)
    detector()