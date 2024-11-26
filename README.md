Home Security System with Real-Time Person Detection and Voice Control
This is a real-time home security system that uses a camera feed to detect human presence, capture screenshots, and allow for remote control via voice commands. The system uses OpenCV, deep learning (MobileNet-SSD), and Flask for a web interface to monitor and control the security status.

Features
Real-Time Human Detection: Detects people using MobileNet-SSD, and marks them with a bounding box.
Arming/Disarming the Security System: Control the security system using voice commands ("Arm Security", "Disarm Security").
Screenshot Capture: Automatically saves screenshots of detected intruders if the system is armed.
Web Interface: View live video feed in a browser, and control the security system from any device.
GPU Support (Optional): Utilizes the CUDA backend (if available) for faster processing on compatible GPUs.
