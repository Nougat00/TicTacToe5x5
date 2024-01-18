# Gesture-based YouTube Player Control

## Overview
This script utilizes the MediaPipe library and OpenCV to perform gesture-based control of a YouTube player using a Decision Tree classifier. It captures webcam footage, detects hand landmarks, and classifies hand gestures to control player actions such as play, pause, forward, and rewind.

## Output
- Real-time visualization of hand landmarks and connections on the webcam feed.
- Gesture-based control of a YouTube player using hand movements.

## Dependencies
- Python libraries: `cv2`, `mediapipe`, `pyautogui`, `time`
- Ensure that the required modules are installed before running the script.

## Dependencies Installation
Make sure to install the required dependencies before running the script. Use the following commands:
```bash
pip install opencv-python
pip install mediapipe
pip install pyautogui
```

## Note
- The script assumes the presence of a single hand in the webcam feed.
- Adjustments to the gesture recognition logic can be made based on individual preferences and requirements.

## Gesture Commands
The script recognizes the following hand gestures for YouTube player control:

1. **One finger (1 Finger):**
   - Fast forward 5 seconds

2. **Two Fingers Gesture (2 Fingers):**
   - Rewind 5 seconds

3. **Three Fingers Gesture (3 Fingers):**
   - Increase volume

4. **Four Fingers Gesture (4 Fingers):**
   - Decrease volume

5. **Five Fingers Gesture (5 Fingers):**
   - Pause/Unpause video

## Usage
1. Run the script using the command: `python main.py`.
2. Ensure the webcam is properly configured and captures your hand gestures.
3. Interact with the YouTube player by making the corresponding hand gestures.