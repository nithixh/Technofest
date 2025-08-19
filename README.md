# Technofest
🎓 Online Exam Proctoring System

An AI-powered proctoring application built with Streamlit, OpenCV, YOLOv8, and DeepFace.
This project was developed for Technofest, where students can explore how artificial intelligence can help monitor online exams securely.

🚀 Features

✅ Face Registration & Login – Students register by capturing their face; login is validated by matching embeddings.
✅ Room Scanning – Uses phone camera (via DroidCam) to scan the environment before the exam. Flags prohibited items (phones, books, extra people).
✅ Exam Monitoring – Live dual-camera monitoring:

Webcam for student’s face

Phone camera for environment
✅ Alerts System – Detects and warns for:

No face detected

Multiple faces detected

Logged-in student not present

Prohibited items (phone, book, extra person)
✅ Lightweight YOLOv8 Model – Real-time object detection
✅ Local Storage (No Cloud) – Faces and data are stored in a simple JSON file (user_db.json), ensuring privacy.

🏗️ Project Structure
technofest-proctoring-app/
├── app/
│   ├── frontend.py    # Streamlit UI (pages, buttons, live feed, alerts)
│   ├── backend.py     # Face recognition & object detection logic
│   ├── database.py    # JSON storage for registered users
│   ├── utils.py       # Helper functions (embeddings, cosine distance)
├── demo.py            # Main entry point & router
├── requirements.txt   # Python dependencies
└── README.md          # Documentation


Frontend (UI Layer) → frontend.py
Handles the interface: login, register, scanning, exam page (via Streamlit).

Backend (Logic Layer) → backend.py
Handles AI/ML logic: YOLO object detection, OpenCV face detection, DeepFace embeddings.

Database (Storage Layer) → database.py
Uses a local JSON file (user_db.json) to store student embeddings.

⚙️ Installation & Setup
1️⃣ Clone Repository
git clone https://github.com/<your-username>/technofest-proctoring-app.git
cd technofest-proctoring-app

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Application
streamlit run demo.py

4️⃣ Access in Browser

Go to: http://localhost:8501

📷 How It Works

Register → Enter name → Capture face (5 images stored as embeddings).

Login → Face recognition validates registered user.

Room Scan → Phone camera scans environment → YOLO detects prohibited items.

Wait Position → Student positions phone camera for full visibility.

Exam Session → Dual monitoring:

Webcam feed ensures only the registered student is present.

Phone feed ensures no cheating (phones/books/extra people).

Alerts appear live in sidebar.

🛠️ Tech Stack

Frontend: Streamlit

Computer Vision: OpenCV
, YOLOv8

Face Recognition: DeepFace

Storage: Local JSON (user_db.json)

🔐 Notes

This project is for educational/demo purposes (Technofest).

No personal data is sent to servers – all processing is local.

For production, a more secure database and authentication system would be required.
