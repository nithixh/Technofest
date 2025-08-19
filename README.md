#Technofest
🎓 Online Exam Proctoring System

An AI-powered online exam monitoring application built with Streamlit, OpenCV, YOLOv8, and DeepFace.
The system ensures fairness and security in remote exams by monitoring both the student and their environment in real time.

🚀 Features

Face Registration & Login – Students register using their face; login is validated by matching embeddings.

Room Scanning – Phone camera scans the environment before the exam to detect prohibited items such as mobile phones, books, or extra people.

Dual Camera Monitoring –

Webcam tracks the student’s face.

Phone camera observes the environment.

Live Alerts System – Automatically detects and notifies if:

No face is visible.

Multiple faces are detected.

The logged-in student is not present.

Prohibited items (phones, books, extra persons) are found.

Lightweight & Real-Time – Uses a YOLOv8 model optimized for fast detection.

Local Data Storage – User data is stored locally in JSON for privacy.

📷 Workflow

Register → Enter your name and capture your face (multiple images).

Login → Face recognition verifies the registered student.

Room Scan → Phone camera scans the environment for prohibited items.

Start Exam →

Webcam ensures the registered student is present.

Phone camera ensures the environment is free from suspicious items.

Alerts are shown live in the sidebar.

🛠️ Technologies Used

Streamlit – User interface

OpenCV – Face detection

DeepFace – Face recognition embeddings

YOLOv8 – Object detection (phones, books, people)

JSON – Local storage for user data
