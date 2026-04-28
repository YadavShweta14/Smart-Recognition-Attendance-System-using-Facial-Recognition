# Smart Recognition Attendance System (SRAS)

An AI-driven, automated attendance management solution built with Python, Flask, and OpenCV.

# Project Overview
The Smart Recognition Attendance System (SRAS) is designed to modernize attendance tracking in educational and corporate environments. It replaces traditional manual methods with a high-speed, touchless facial recognition pipeline that ensures data integrity and prevents proxy attendance.

# Key Features
* **Real-time Recognition:** Sub-millisecond face detection using **MediaPipe**.
* **LBPH Algorithm:** Robust identification using Local Binary Patterns Histograms via OpenCV.
* **Anti-Duplicate Logic:** Intelligent verification ensures students are marked "Present" only once per day.
* **Admin Dashboard:** Flask-based web interface for live monitoring, registration, and log management.
* **Visual Feedback:** Displays real-time status messages like "Already Recorded" on the live camera feed.

# Technology Stack
| Component | Technology |

| **Language** | Python 3.x |
| **Web Framework** | Flask |
| **Face Detection** | MediaPipe |
| **Face Recognition** | OpenCV (LBPH) |
| **Data Handling** | Pandas, NumPy |

# Project Structure
SRAS/
├── app.py              # Main application logic
├── dataset/            # Student facial images (organized by ID)
├── static/             # UI styling (CSS/JS)
├── templates/          # HTML Dashboard
├── attendance.csv      # Daily attendance logs
├── students.csv        # Student master records
└── face_model.yml      # Trained recognition model

# Installation
Clone the repository:
git clone [https://github.com/yourusername/sras-project.git](https://github.com/yourusername/sras-project.git)
cd sras-project

# Install dependencies:
pip install flask opencv-contrib-python mediapipe pandas numpy

# Run the app:
python app.py
Access the dashboard at http://127.0.0.1:5000.

# Usage
Register: Input student details and capture 100 face samples.
Train: The system automatically generates/updates the face_model.yml.
Recognize: Launch the live feed. The system identifies faces and checks the database for existing records before logging "Present".

# Security Note
This project uses a .gitignore file to ensure that private student data (images and logs) is not uploaded to public repositories.

Developed by Shweta Yadav
BSc Computer Science




