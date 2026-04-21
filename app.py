from flask import Flask, render_template, request, redirect, Response, url_for
import cv2
import os
import numpy as np
import pandas as pd
import mediapipe as mp
from datetime import datetime, date

app = Flask(__name__)

# --- GLOBAL CONTROLS (Crucial for Capture) ---
capture_count = 0
current_registering_id = None

# --- CONFIGURATION ---
DATASET = "dataset/student_images"
MODEL_PATH = "models/face_model.yml"
STUDENTS_CSV = "data/students.csv"
ATTENDANCE_CSV = "data/attendance.csv"

# Ensure all folders exist
for folder in ["data", "models", DATASET]:
    os.makedirs(folder, exist_ok=True)

# --- INITIALIZE DATABASE (Using Tabs \t) ---
def init_csv():
    if not os.path.exists(STUDENTS_CSV):
        pd.DataFrame(columns=["id", "name", "roll", "course", "year"]).to_csv(STUDENTS_CSV, index=False)
    if not os.path.exists(ATTENDANCE_CSV):
        pd.DataFrame(columns=["id", "date", "time", "status"]).to_csv(ATTENDANCE_CSV, index=False)

init_csv()

# --- FACE RECOGNITION SETUP ---
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)
recognizer = cv2.face.LBPHFaceRecognizer_create()

if os.path.exists(MODEL_PATH):
    try:
        recognizer.read(MODEL_PATH)
    except:
        print("Model file empty or corrupt.")

# --- LOGIC FUNCTIONS ---

def train_model():
    global recognizer # Global recognizer ko use karein
    faces, labels = [], []
    if not os.path.exists(DATASET): return
    for sid in os.listdir(DATASET):
        folder = os.path.join(DATASET, sid)
        if not os.path.isdir(folder): continue
        for img_name in os.listdir(folder):
            path = os.path.join(folder, img_name)
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if gray is not None:
                faces.append(cv2.resize(gray, (200, 200)))
                labels.append(int(sid))
    if faces:
        recognizer.train(faces, np.array(labels))
        recognizer.save(MODEL_PATH)
        # --- SABSE ZAROORI LINE ---
        recognizer.read(MODEL_PATH) # Nayi training ko memory mein load karein
        print("AI Model Updated, Saved and Reloaded into Memory.")

def detect_and_recognize(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)
    recognized = []
    if results.detections:
        h, w, _ = frame.shape
        for det in results.detections:
            box = det.location_data.relative_bounding_box
            x, y = max(0, int(box.xmin * w)), max(0, int(box.ymin * h))
            bw, bh = int(box.width * w), int(box.height * h)
            face_roi = frame[y:y+bh, x:x+bw]
            if face_roi.size > 0:
                gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                gray_face = cv2.resize(gray_face, (200, 200))
                try:
                    label, confidence = recognizer.predict(gray_face)
                    # DEBUG PRINT: Terminal mein check karein ye kya dikha raha hai
                    #print(f"Checking Face: ID {label} with Confidence {confidence}")
                    
                    # Threshold ko 100 kar dein testing ke liye
                    if confidence < 100: 
                        recognized.append((x, y, bw, bh, label))
                except Exception as e: 
                    print(f"Prediction Error: {e}")
    return recognized
"""
def train_model():
    faces, labels = [], []
    if not os.path.exists(DATASET): return
    for sid in os.listdir(DATASET):
        folder = os.path.join(DATASET, sid)
        if not os.path.isdir(folder): continue
        for img_name in os.listdir(folder):
            path = os.path.join(folder, img_name)
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if gray is not None:
                # Resize to ensure consistent LBPH training
                faces.append(cv2.resize(gray, (200, 200)))
                labels.append(int(sid))
    if faces:
        recognizer.train(faces, np.array(labels))
        recognizer.save(MODEL_PATH)
        print("AI Model Updated and Saved.")

def detect_and_recognize(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)
    recognized = []
    if results.detections:
        h, w, _ = frame.shape
        for det in results.detections:
            box = det.location_data.relative_bounding_box
            x, y = max(0, int(box.xmin * w)), max(0, int(box.ymin * h))
            bw, bh = int(box.width * w), int(box.height * h)
            face_roi = frame[y:y+bh, x:x+bw]
            if face_roi.size > 0:
                gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                gray_face = cv2.resize(gray_face, (200, 200))
                try:
                    label, confidence = recognizer.predict(gray_face)
                    if confidence < 80: # Lower is more confident
                        recognized.append((x, y, bw, bh, label))
                except: pass
    return recognized
"""
# --- ROUTES ---

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = request.form.get('username')
        if user == 'admin':
            return redirect(url_for('dashboard'))
        return "Invalid Credentials"
    return render_template('login.html')

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/register", methods=["POST"])
def register():
    global current_registering_id, capture_count
    try:
        # Get raw data from the empty text fields
        name = request.form.get("name")
        roll = request.form.get("roll")
        course = request.form.get("course")
        year = request.form.get("year")
        
        # Load existing data with tabs
        df = pd.read_csv(STUDENTS_CSV)
        
        # Create a simple numeric ID based on count
        sid = len(df) + 101 
        
        # Safe append to prevent column mismatch
        new_entry = pd.DataFrame([[sid, name, roll, course, year]], 
                                 columns=["id", "name", "roll", "course", "year"])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(STUDENTS_CSV, index=False)

        
        
        # Prepare image folder
        os.makedirs(os.path.join(DATASET, str(sid)), exist_ok=True)
        
        # Trigger the camera to start saving frames
        current_registering_id = sid
        capture_count = 0
        
        print(f"Student {name} registered with ID {sid}. Starting capture...")
        return redirect(url_for('dashboard'))
    except Exception as e:
        print(f"Registration Error: {e}")
        return redirect(url_for('dashboard'))
    
  
# Add this to your globals at the top
feedback_counter = 0
feedback_text = ""

def gen_frames():
    global capture_count, current_registering_id, feedback_counter, feedback_text
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success: 
            break
        
        # --- 1. CAPTURE & REGISTRATION FEEDBACK ---
        if current_registering_id is not None and capture_count < 1:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            folder = os.path.join(DATASET, str(current_registering_id))
            cv2.imwrite(f"{folder}/face.jpg", cv2.resize(gray, (200, 200)))
            
            capture_count = 1 
            train_model()
            
            # Set the persistent message
            feedback_text = "REGISTRATION SUCCESSFUL!"
            feedback_counter = 50  # Stay on screen for ~40 frames
            
            current_registering_id = None 

        # --- 2. DETECTION & ATTENDANCE ---
        results = detect_and_recognize(frame)
        for (x, y, w, h, label) in results:
            att_df = pd.read_csv(ATTENDANCE_CSV)
            today = str(date.today())
            
            is_present = ((att_df["id"].astype(str) == str(label)) & (att_df["date"] == today)).any()
            
            if not is_present:
                now = datetime.now().strftime("%H:%M:%S")
                new_att = pd.DataFrame([[label, today, now, "Present"]], 
                                       columns=["id", "date", "time", "status"])
                att_df = pd.concat([att_df, new_att], ignore_index=True)
                att_df.to_csv(ATTENDANCE_CSV, index=False)
                
                # Set feedback for new attendance
                feedback_text = f"ID: {label} - MARKED PRESENT"
                feedback_counter = 50
                color = (0, 255, 0)
            else:
                # Already marked - just show the label box
                cv2.putText(frame, "PRESENT", (x, y-30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 191, 0), 1)
                color = (255, 191, 0)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"ID: {label}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # --- 3. RENDER PERSISTENT FEEDBACK OVERLAY ---
        if feedback_counter > 0:
            # Draw a background rectangle for the text (makes it easier to read)
            cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1) 
            cv2.putText(frame, feedback_text, (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            feedback_counter -= 1 # Reduce the timer by 1 each frame

        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    cap.release()

@app.route("/view_attendance")
def view_attendance():
    try:
        if os.path.exists(ATTENDANCE_CSV):
            df = pd.read_csv(ATTENDANCE_CSV)
            if not df.empty:
                # Table ka design set kar rahe hain
                attendance_html = df.to_html(classes='table table-bordered table-hover', index=False)
            else:
                attendance_html = "<div class='alert alert-warning'>Abhi tak koi attendance record nahi hai.</div>"
        else:
            attendance_html = "<div class='alert alert-danger'>Attendance file (CSV) nahi mili!</div>"
            
        return f"""
        <html>
            <head>
                <title>Attendance Data</title>
                <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
                <style>
                    body {{ background-color: #f8f9fa; padding-top: 50px; }}
                    .container {{ background: white; padding: 30px; border-radius: 15px; box-shadow: 0px 0px 15px rgba(0,0,0,0.1); }}
                    h2 {{ color: #333; font-weight: bold; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h2 class="mb-4 text-center">Live Attendance Records</h2>
                    <hr>
                    {attendance_html}
                    <div class="text-center mt-4">
                        <a href="/dashboard" class="btn btn-dark">Back to Dashboard</a>
                    </div>
                </div>
            </body>
        </html>
        """
    except Exception as e:
        return f"System Error: {e}"


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True, port=5000)