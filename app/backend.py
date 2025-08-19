import cv2
from ultralytics import YOLO
from app.utils import embed_face, cosine_dist

YOLO_MODEL = "yolov8n.pt"

def load_yolo():
    try:
        return YOLO(YOLO_MODEL)
    except Exception:
        return None

def detect_face_opencv(frame):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    face_detected = len(faces) > 0
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return frame, face_detected, len(faces)

def detect_objects(frame, yolo_model, prohibited_items):
    if yolo_model is None:
        return frame, []
    results = yolo_model(frame, verbose=False)
    detected_items = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = yolo_model.names[class_id]
            conf = float(box.conf[0])
            if class_name in prohibited_items and conf > 0.5:
                detected_items.append((class_name, conf))
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return frame, detected_items
