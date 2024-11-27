import cv2
import numpy as np
import threading
import time
import os
import json
from deepface import DeepFace
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Deaktiver oneDNN-optimeringer
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Skjul TensorFlow-advarsler


# Define a class to track person IDs
class PersonTracker:
    def __init__(self):
        self.face_id = 0
        self.person_ids = {}
        self.saved_ids = set()
        self.saved_ids_raw = set()

    def get_person_id(self):
        self.face_id += 1
        return self.face_id

    def update_person_id(self, face_id, person_id):
        self.person_ids[face_id] = person_id

    def get_person_id_from_face_id(self, face_id):
        return self.person_ids.get(face_id)

    def is_person_id_saved(self, person_id):
        return person_id in self.saved_ids

    def is_person_id_saved_raw(self, person_id):
        return person_id in self.saved_ids_raw

    def mark_person_id_as_saved(self, person_id):
        self.saved_ids.add(person_id)

    def mark_person_id_as_saved_raw(self, person_id):
        self.saved_ids_raw.add(person_id)

    def get_next_unique_person_id(self):
        while True:
            new_person_id = self.get_person_id()
            if not self.is_person_id_saved(new_person_id):
                return new_person_id

# Function to brighten the face region
def apply_studio_light(frame, startX, startY, endX, endY, brightness=2):
    # Extract the face region
    face = frame[startY:endY, startX:endX]

    # Apply brightness effect using convertScaleAbs
    face = cv2.convertScaleAbs(face, alpha=brightness, beta=0)

    # Place the brightened face region back into the frame
    frame[startY:endY, startX:endX] = face

# Function to guess race/ethnicity (basic heuristic for demo purposes)
def guess_race_attributes(image, startX, startY, endX, endY):
    # Extract the region of interest (ROI)
    roi = image[startY:endY, startX:endX]

        # Convert ROI to a format compatible with DeepFace (if necessary)
    # DeepFace expects images in RGB format
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    # Analyze the face using DeepFace
    analysis = DeepFace.analyze(roi_rgb, actions=['age', 'gender', 'race', 'emotion'])

    # Extract values from the analysis
    age = analysis[0]['age']
    gender = analysis[0]['dominant_gender']
    dominant_race = analysis[0]['dominant_race']
    dominant_emotion = analysis[0]['dominant_emotion']

    return [age, gender, dominant_race, dominant_emotion]

# Function to analyze the image and save the output to a JSON file
def analyze_and_save_json(image_path, json_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert image to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Analyze the face using DeepFace
    try:
        analysis = DeepFace.analyze(image_rgb)
        
        # Save the analysis to a JSON file
        with open(json_path, 'w') as json_file:
            json.dump(analysis, json_file)
    except ValueError as e:
        print(f"Error analyzing face: {e}")

def save_detected_face(frame, startX, startY, endX, endY, person_id):
    if not os.path.exists("detected_faces"):
        os.makedirs("detected_faces")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"detected_faces/face_{person_id}_{timestamp}.jpg"

    cv2.imwrite(filename, frame[startY:endY, startX:endX])

    person_tracker.mark_person_id_as_saved(person_id)

def save_detected_face_raw(frame, startX, startY, endX, endY, person_id):
    if not os.path.exists("detected_faces"):
        os.makedirs("detected_faces")

    apply_studio_light(frame, startX-100, startY-100, endX+100, endY+100, 1.5)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"detected_faces/face_{person_id}_{timestamp}_raw.jpg"
    json_filename = f"detected_faces/face_{person_id}_{timestamp}_raw.json"

    cv2.imwrite(filename, frame[startY-100:endY+100, startX-100:endX+100])

    # Start a new thread to analyze the image and save the output to a JSON file
    analysis_thread = threading.Thread(target=analyze_and_save_json, args=(filename, json_filename))
    analysis_thread.start()

    person_tracker.mark_person_id_as_saved_raw(person_id)

def detect_faces(frame, net, min_confidence, person_tracker):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    max_confidence = 0

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > min_confidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face_id = i
            person_id = person_tracker.get_person_id_from_face_id(face_id)
            if person_id is None:
                person_id = person_tracker.get_next_unique_person_id()
                person_tracker.update_person_id(face_id, person_id)

            # Save the raw detected face
            if not person_tracker.is_person_id_saved_raw(person_id):
                savedFrame = frame.copy()
                save_detected_face_raw(savedFrame, startX, startY, endX, endY, person_id)

            # Draw the bounding box around the face
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

            text = f"Person ID: {person_id}"
            cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            if confidence > max_confidence:
                max_confidence = confidence

            # Apply the brightening effect to the detected face
            apply_studio_light(frame, startX, startY, endX, endY, 2)

            # Save the detected face
            if not person_tracker.is_person_id_saved(person_id):
                save_detected_face(frame, startX, startY, endX, endY, person_id)


            # Guess face attibutes
            attributes = guess_race_attributes(frame, startX, startY, endX, endY)

            # Display the race/ethnicity guess
            ethnicity_text = f"Race/Ethnicity: {attributes[2]}"
            cv2.putText(frame, ethnicity_text, (startX, startY + (endY - startY) + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Display the age guess
            ethnicity_text = f"Age: {attributes[0]}"
            cv2.putText(frame, ethnicity_text, (startX, startY + (endY - startY) + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Display the gender guess
            ethnicity_text = f"Gender: {attributes[1]}"
            cv2.putText(frame, ethnicity_text, (startX, startY + (endY - startY) + 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return max_confidence

def process_frame(frame, net, min_confidence, person_tracker, start_time):
    confidence = detect_faces(frame, net, min_confidence, person_tracker)

    end_time = time.time()
    fps = 1 / (end_time - start_time)

    # Smaller font size for FPS and confidence
    font_scale = 0.5
    font_thickness = 1

    if fps < 20:
        fps_color = (0, 0, 255)  # Red
    elif 20 <= fps <= 50:
        fps_color = (0, 165, 255)  # Orange
    else:
        fps_color = (0, 255, 0)  # Green

    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (18, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, fps_color, font_thickness)

    confidence_text = f"Confidence: {confidence:.2f}"
    cv2.putText(frame, confidence_text, (18, 80), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness)

# Load the face detection model
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

min_confidence = 0.35
cap = cv2.VideoCapture(0)

person_tracker = PersonTracker()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    start_time = time.time()
    thread = threading.Thread(target=process_frame, args=(frame, net, min_confidence, person_tracker, start_time))

    thread.start()
    thread.join()

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()