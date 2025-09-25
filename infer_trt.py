import cv2
from runtime.jetson.trt_detector import TRTDetector
from runtime.common.email_alerts import send_email_alert

def run():
    detector = TRTDetector("exports/best_fp16.engine", conf=0.35)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret: break

        detections = detector(frame)
        labels = ["smoke" if int(cls)==0 else "fire" for *_, conf, cls in detections]

        if labels:
            print("[DETECTION]", labels)
            send_email_alert(labels)

        cv2.imshow("Jetson Wildfire", frame)
        if cv2.waitKey(1) == 27: break
