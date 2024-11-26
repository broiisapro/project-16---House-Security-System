import cv2
import os
import time
from datetime import datetime
import threading
import speech_recognition as sr
from flask import Flask, render_template, Response, jsonify

# Security state to manage arming/disarming
security_state = {"armed": True}

# Directory for saving screenshots
os.makedirs("captures", exist_ok=True)

# Load the pre-trained MobileNet-SSD model for GPU-based processing
def load_model():
    model_config = "MobileNetSSD_deploy.prototxt"
    model_weights = "MobileNetSSD_deploy.caffemodel"
    net = cv2.dnn.readNetFromCaffe(model_config, model_weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)  # Use CUDA backend
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)  # Use GPU
    return net

# Detect humans in the video frame
def detect_humans(frame, net, confidence_threshold=0.5):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    humans = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            class_id = int(detections[0, 0, i, 1])
            if class_id == 15:  # Class ID 15 corresponds to "person"
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (start_x, start_y, end_x, end_y) = box.astype("int")
                humans.append((start_x, start_y, end_x, end_y))
    return humans

# Save a screenshot of the current frame
def save_screenshot(frame):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    image_path = f"captures/human_detected_{timestamp}.jpg"
    cv2.imwrite(image_path, frame)
    return image_path

# Listen for voice commands to arm/disarm the system
def listen_for_commands():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for 'Arm Security' or 'Disarm Security' commands...")
        while True:
            try:
                audio = recognizer.listen(source, timeout=5)
                command = recognizer.recognize_google(audio).lower()
                if "arm security" in command:
                    security_state["armed"] = True
                    print("Security system armed!")
                elif "disarm security" in command:
                    security_state["armed"] = False
                    print("Security system disarmed!")
            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                pass

# Flask Web Server Setup
app = Flask(__name__)

# Initialize camera
camera = cv2.VideoCapture(0)  # Adjust index for your camera
net = load_model()

# Global variable to store last screenshot time
last_screenshot_time = time.time()

# Video streaming function for Flask
def generate_video():
    global last_screenshot_time  # Declare as global to modify it

    while True:
        ret, frame = camera.read()
        if not ret:
            continue

        # Detect humans in the frame
        humans = detect_humans(frame, net)

        # Draw bounding boxes for detected humans
        for (start_x, start_y, end_x, end_y) in humans:
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            cv2.putText(frame, "Human", (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Save screenshots and log events if armed
        if humans and security_state["armed"] and time.time() - last_screenshot_time >= 5:
            image_path = save_screenshot(frame)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Human detected! Screenshot saved at {image_path}")
            last_screenshot_time = time.time()

        # Convert frame to JPEG and stream it to the web browser
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Web route to stream video
@app.route('/video_feed')
def video_feed():
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Web route to control arm/disarm
@app.route('/arm_security')
def arm_security():
    security_state["armed"] = True
    return jsonify({"status": "Security system armed!"})

@app.route('/disarm_security')
def disarm_security():
    security_state["armed"] = False
    return jsonify({"status": "Security system disarmed!"})

# Web route to display dashboard
@app.route('/')
def index():
    return render_template('index.html')

# Start Flask app in a separate thread for web interface
def run_flask():
    app.run(host="0.0.0.0", port=5000, threaded=True)

# Main function to process video feed
def main():
    # Start the Flask web server
    threading.Thread(target=run_flask, daemon=True).start()

    # Start the voice command listener in a separate thread
    threading.Thread(target=listen_for_commands, daemon=True).start()

    print("Starting human detection...")

    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
