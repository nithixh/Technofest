# Requirements: streamlit, opencv-python, deepface, ultralytics, numpy, pillow

import streamlit as st
import cv2
import json
import os
import numpy as np
from deepface import DeepFace
from ultralytics import YOLO
import time
from threading import Thread

class ThreadedVideoCapture:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.ret = False
        self.frame = None
        self.running = True
        self.thread = None
        
        if self.cap.isOpened():
            self.thread = Thread(target=self.update, daemon=True)
            self.thread.start()
        else:
            self.running = False

    def isOpened(self):
        return self.cap.isOpened()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.ret = True
            self.frame = frame
        # cleanup when loop ends
        if self.cap.isOpened():
            self.cap.release()

    def read(self):
        return self.ret, self.frame

    def release(self):
        """Stop the thread and close the camera cleanly."""
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2)  # Wait for update() to exit
        if self.cap.isOpened():
            self.cap.release()



# ---------------------------  CONFIG  ---------------------------------
DB_PATH = "user_db.json"        # JSON file holding registered users
PROHIBITED_ITEMS_SCAN = ["cell phone", "book"]  # During room scan - any person is suspicious
PROHIBITED_ITEMS_EXAM = ["cell phone", "book"]  # During exam - student is expected to be visible
YOLO_MODEL = "yolov8n.pt"       # light-weight model (auto-downloads)
DIST_THRESHOLD = 0.4            # cosine distance threshold for face matching
PHONE_URL_DEFAULT = "http://192.168.218.73:4747/video"  # DroidCam default

# ---------------------------  HELPERS  --------------------------------

def load_database():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "r") as f:
            return json.load(f)
    return []

def save_database(db):
    with open(DB_PATH, "w") as f:
        json.dump(db, f, indent=2)

@st.cache_resource(show_spinner=False)
def load_yolo():
    try:
        return YOLO(YOLO_MODEL)
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}")
        return None

def get_webcam(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        st.error("❌ Webcam not accessible")
        return None
    return cap

# DeepFace embeds a single frame to 128-D vector (Facenet default)
def embed_face(img_bgr):
    try:
        # DeepFace expects BGR image
        embedding = DeepFace.represent(img_path=img_bgr, model_name="Facenet", enforce_detection=False)[0]["embedding"]
        return np.array(embedding)
    except Exception as e:
        return None

# Cosine similarity
def cosine_dist(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return 1 - np.dot(a, b)

# Face detection using OpenCV (backup method)
def detect_face_opencv(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    face_detected = len(faces) > 0
    
    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Face Detected", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame, face_detected, len(faces)

# Object detection function
def detect_objects(frame, yolo_model, prohibited_items, is_exam_mode=False):
    if yolo_model is None:
        return frame, []
    
    try:
        results = yolo_model(frame, verbose=False)
        detected_items = []
        person_count = 0
        laptop_count = 0
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = yolo_model.names[class_id]
                    confidence = float(box.conf[0])

                    # Count persons & laptops separately
                    if class_name == "person" and confidence > 0.5:
                        person_count += 1
                    if class_name == "laptop" and confidence > 0.5:
                        laptop_count += 1

                    # Flag other prohibited items (phone, book, etc.)
                    if class_name in prohibited_items and confidence > 0.5:
                        detected_items.append((class_name, confidence))
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"PROHIBITED: {class_name}",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 0, 255), 2)

        # After counting all objects, flag extra persons/laptops
        if person_count > 1:
            detected_items.append(("Extra Person", 1.0))
        if laptop_count > 1:
            detected_items.append(("Extra Laptop", 1.0))

        return frame, detected_items
    except Exception:
        return frame, []


# ---------------------------  APP STATE  ------------------------------
if "phase" not in st.session_state:
    st.session_state.phase = "landing"   # landing → register → login → scan → wait_position → exam
    st.session_state.username = ""
    st.session_state.db = load_database()
    st.session_state.phone_url = PHONE_URL_DEFAULT
    st.session_state.yolo = load_yolo()
    st.session_state.alerts = []
    st.session_state.monitoring = False
    st.session_state.scan_complete = False

# ---------------------------  LANDING  --------------------------------

def landing_page():
    st.title("🎓 Online Exam Proctoring System")
    st.subheader("Choose an option")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📝 Register (Sign-Up)", key="register_btn"):
            st.session_state.phase = "register"
            st.rerun()
    
    with col2:
        if st.button("🔑 Login & Start Exam", key="login_btn"):
            st.session_state.phase = "login"
            st.rerun()

# ---------------------------  REGISTRATION  ---------------------------

def register_page():
    st.title("📝 Face Registration")
    
    name = st.text_input("Enter your full name")
    
    col1, col2 = st.columns(2)
    with col1:
        capture_btn = st.button("📷 Start Camera & Capture", key="capture_btn")
    with col2:
        back_btn = st.button("⬅️ Back", key="back_from_register")
    
    if back_btn:
        st.session_state.phase = "landing"
        st.rerun()
        return
    
    if not capture_btn or name.strip() == "":
        st.info("Enter name and click capture")
        return
    
    cap = get_webcam(0)
    if cap is None:
        return
    
    st.write("Show your face clearly – capturing 5 images…")
    embeddings = []
    placeholder = st.empty()
    
    for i in range(5):
        ret, frame = cap.read()
        if not ret:
            continue
        
        emb = embed_face(frame)
        if emb is not None:
            embeddings.append(emb.tolist())
        
        frame = cv2.putText(frame, f"Captured {i+1}/5", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        time.sleep(1)
    
    cap.release()
    
    if len(embeddings) < 3:
        st.error("Face capture failed – try again with better lighting")
        return
    
    # Save to DB
    st.session_state.db.append({"name": name, "embeddings": embeddings})
    save_database(st.session_state.db)
    st.success(f"✅ Registered {name}!")
    
    if st.button("Proceed to Login", key="proceed_to_login"):
        st.session_state.phase = "login"
        st.rerun()

# ---------------------------  LOGIN  ----------------------------------

def login_page():
    st.title("🔑 Face Login")
    
    col1, col2 = st.columns(2)
    with col1:
        login_btn = st.button("📷 Start Camera & Login", key="start_login")
    with col2:
        back_btn = st.button("⬅️ Back", key="back_from_login")
    
    if back_btn:
        st.session_state.phase = "landing"
        st.rerun()
        return
    
    if not login_btn:
        return
    
    if len(st.session_state.db) == 0:
        st.warning("No users registered – please sign-up first")
        return
    
    cap = get_webcam(0)
    if cap is None:
        return
    
    placeholder = st.empty()
    matched_name = None
    start_time = time.time()
    timeout = 10  # 10-sec window
    
    while time.time() - start_time < timeout and matched_name is None:
        ret, frame = cap.read()
        if not ret:
            continue
        
        emb = embed_face(frame)
        if emb is not None:
            for user in st.session_state.db:
                dists = [cosine_dist(emb, np.array(e)) for e in user["embeddings"]]
                if np.mean(dists) < DIST_THRESHOLD:
                    matched_name = user["name"]
                    break
        
        frame_disp = frame.copy()
        txt = matched_name if matched_name else "Recognising…"
        cv2.putText(frame_disp, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        placeholder.image(cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB), channels="RGB")
    
    cap.release()
    
    if matched_name:
        st.success(f"✅ Welcome {matched_name}!")
        st.session_state.username = matched_name
        st.session_state.phase = "scan"
        time.sleep(2)
        st.rerun()
    else:
        st.error("Face not recognised – try again")

# ---------------------------  SCANNING  -------------------------------

def scanning_page():
    st.title("📱 Room Scanning – Phone Camera Only")
    
    phone_url = st.text_input("Phone camera URL (DroidCam)", st.session_state.phone_url)
    st.session_state.phone_url = phone_url
    
    st.info("📱 During scanning, move your phone around the room to show all areas. Any person detected will be flagged as suspicious since you should be holding the phone.")
    
    col1, col2 = st.columns(2)
    with col1:
        start_scan = st.button("🔍 Begin Room Scan", key="start_scan_btn")
    with col2:
        skip_scan = st.button("⏭️ Skip Scan (for demo)", key="skip_scan_btn")
    
    if skip_scan:
        st.session_state.scan_complete = True
        st.session_state.phase = "wait_position"
        st.rerun()
        return
    
    if not start_scan:
        st.info("Place phone camera to show entire room, then click begin")
        return
    
    # Connect to phone cam
    cap=ThreadedVideoCapture(st.session_state.phone_url)
    if not cap.isOpened():
        st.error("Cannot connect to phone camera – check URL & WiFi")
        return
    
    st.info("🔍 Scanning room – move phone to show all areas…")
    
    placeholder = st.empty()
    detected_items = []
    scan_duration = 20  # 10 seconds of scanning
    start_time = time.time()
    
    # Create stop button outside the loop
    stop_placeholder = st.empty()
    
    while time.time() - start_time < scan_duration:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Object detection for scanning (includes person detection)
        frame_disp, current_detections = detect_objects(frame, st.session_state.yolo, 
                                                       PROHIBITED_ITEMS_SCAN, is_exam_mode=False)
        
        # Add current detections to overall list
        for item in current_detections:
            if item not in detected_items:
                detected_items.append(item)
        
        # Show remaining time
        remaining = scan_duration - (time.time() - start_time)
        cv2.putText(frame_disp, f"Scanning: {remaining:.1f}s", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        placeholder.image(cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB), channels="RGB")
        
        # Check if stop button is pressed
        if stop_placeholder.button("⏹️ Stop Early", key=f"stop_scan_{int(time.time()*1000)}"):
            break
        
        time.sleep(0.1)
    
    cap.release()
    stop_placeholder.empty()
    
    if detected_items:
        st.error("🚫 Prohibited items detected:")
        for item, conf in detected_items:
            st.warning(f"- {item} (confidence: {conf:.2f})")
        st.error("Please remove these items and scan again")
    else:
        st.success("✅ Room scan complete - no prohibited items detected!")
        st.session_state.scan_complete = True
        if st.button("➡️ Proceed to Camera Positioning", key="proceed_after_scan"):
            st.session_state.phase = "wait_position"
            st.rerun()

# ---------------------------  WAIT PHONE POSITION ---------------------

def wait_position_page():
    st.title("📱 Position Phone Camera & Start Exam")
    
    st.info("📍 Position your phone camera so that:")
    st.markdown("""
    - Your desk and workspace are visible
    - You can be seen in the frame (this is normal and expected)
    - The camera has a clear view of your hands and materials
    - The phone is stable and won't move during the exam
    """)
    
    st.warning("⚠️ Note: The system expects you to be visible in the phone camera during the exam. Only additional people will trigger alerts.")
    
    if st.button("✅ START EXAM", key="start_exam_btn"):
        st.session_state.phase = "exam"
        st.session_state.monitoring = True
        st.rerun()

# ---------------------------  EXAM MONITORING -------------------------

def exam_page():
    st.title(f"🖥️ Exam Session – {st.session_state.username}")
    
    # Control buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("⏹️ End Exam & Logout", key="end_exam"):
            if "webcam" in st.session_state and st.session_state.webcam:
                st.session_state.webcam.release()
                st.session_state.webcam = None
            if "phone" in st.session_state and st.session_state.phone:
                st.session_state.phone.release()
                st.session_state.phone = None
            st.session_state.phase = "landing"
            st.session_state.monitoring = False
            st.session_state.alerts = []
            st.session_state.scan_complete = False
            st.rerun()

    with col2:
        if st.button("⏸️ Pause Monitoring", key="pause_monitoring"):
            if "webcam" in st.session_state and st.session_state.webcam:
                st.session_state.webcam.release()
                st.session_state.webcam = None
            if "phone" in st.session_state and st.session_state.phone:
                st.session_state.phone.release()
                st.session_state.phone = None
            st.session_state.monitoring = False
            st.rerun()

    with col3:
        if st.button("▶️ Resume Monitoring", key="resume_monitoring"):
            st.session_state.monitoring = True
            st.rerun()

    # Status display
    if st.session_state.monitoring:
        st.success("🟢 MONITORING ACTIVE")
    else:
        st.warning("⏸️ MONITORING PAUSED")
    
    # Camera feeds
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📹 Webcam Feed (Face Monitoring)")
        webcam_placeholder = st.empty()
    with col2:
        st.subheader("📱 Phone Camera Feed (Environment)")
        phone_placeholder = st.empty()
    
    # Alerts sidebar placeholder (for live updates)
    alerts_placeholder = st.sidebar.empty()

    # Start monitoring if active
    if st.session_state.monitoring:
        # Setup cameras
        if "webcam" not in st.session_state or st.session_state.webcam is None:
            st.session_state.webcam = ThreadedVideoCapture(0)
        if "phone" not in st.session_state or st.session_state.phone is None:
            st.session_state.phone = ThreadedVideoCapture(st.session_state.phone_url)
            #st.session_state.phone.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        webcam = st.session_state.webcam
        phone = st.session_state.phone
        
        if not phone.isOpened():
            st.error("❌ Phone camera not accessible - check DroidCam connection")
            return
        
        # Monitoring loop
        for i in range(100):
            if not st.session_state.monitoring:
                break
            
            # Webcam feed
            ret1, frame1 = webcam.read()
            if ret1:
                emb = embed_face(frame1)
                matched_person = False
                frame1_disp, face_detected, face_count = detect_face_opencv(frame1.copy())

                if not face_detected:
                    alert = f"⚠️ No face detected - {time.strftime('%H:%M:%S')}"
                    if alert not in st.session_state.alerts:
                        st.session_state.alerts.append(alert)
                elif face_count > 1:
                    alert = f"⚠️ Multiple faces detected - {time.strftime('%H:%M:%S')}"
                    if alert not in st.session_state.alerts:
                        st.session_state.alerts.append(alert)

                if emb is not None:
                    for user in st.session_state.db:
                        if user["name"] == st.session_state.username:
                            dists = [cosine_dist(emb, np.array(e)) for e in user["embeddings"]]
                            if np.mean(dists) < DIST_THRESHOLD:
                                matched_person = True
                                break

                if not matched_person:
                    alert = f"🚨 Logged-in person not detected - {time.strftime('%H:%M:%S')}"
                    if alert not in st.session_state.alerts:
                        st.session_state.alerts.append(alert)

                if matched_person:
                    cv2.putText(frame1_disp, "Verified", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if face_count > 1:
                        cv2.putText(frame1_disp, "Multiple Faces!", (200, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(frame1_disp, "Not Verified", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if face_count > 1:
                        cv2.putText(frame1_disp, "Multiple Faces!", (200, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                frame1_rgb = cv2.cvtColor(frame1_disp, cv2.COLOR_BGR2RGB)
                webcam_placeholder.image(frame1_rgb, channels="RGB", use_container_width=True)

            # Phone feed
            ret2, frame2 = phone.read()
            if ret2:
                frame2_disp, objects = detect_objects(frame2.copy(), st.session_state.yolo, 
                                                    PROHIBITED_ITEMS_EXAM, is_exam_mode=True)
                
                for obj_name, conf in objects:
                    alert = f"🚫 {obj_name} detected - {time.strftime('%H:%M:%S')}"
                    if alert not in st.session_state.alerts:
                        st.session_state.alerts.append(alert)

                frame2_rgb = cv2.cvtColor(frame2_disp, cv2.COLOR_BGR2RGB)
                phone_placeholder.image(frame2_rgb, channels="RGB", use_container_width=True)

            # Update live alerts in sidebar
            with alerts_placeholder.container():
                st.header("🚨 Live Alerts")
                if st.session_state.alerts:
                    for alert in st.session_state.alerts[-10:]:
                        st.warning(alert)
                else:
                    st.success("No alerts detected")
                if st.button("🗑️ Clear Alerts", key=f"clear_alerts_{i}"):
                    st.session_state.alerts = []

            time.sleep(0.1)
    
    else:
        webcam_placeholder.info("📹 Webcam feed paused")
        phone_placeholder.info("📱 Phone camera feed paused")
        with alerts_placeholder.container():
            st.header("🚨 Live Alerts")
            if st.session_state.alerts:
                for alert in st.session_state.alerts[-10:]:
                    st.warning(alert)
            else:
                st.success("No alerts detected")
            if st.button("🗑️ Clear Alerts", key="clear_alerts_paused"):
                st.session_state.alerts = []



# ---------------------------  MAIN ROUTER -----------------------------

def main():
    # Set page config
    st.set_page_config(
        page_title="Online Exam Proctoring System",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Route to appropriate page
    if st.session_state.phase == "landing":
        landing_page()
    elif st.session_state.phase == "register":
        register_page()
    elif st.session_state.phase == "login":
        login_page()
    elif st.session_state.phase == "scan":
        scanning_page()
    elif st.session_state.phase == "wait_position":
        wait_position_page()
    elif st.session_state.phase == "exam":
        exam_page()

if __name__ == "__main__":
    main()
