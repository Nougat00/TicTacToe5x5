"""
This script utilizes the MediaPipe library and OpenCV to perform gesture-based control of a YouTube player using a Decision Tree classifier.
It captures webcam footage, detects hand landmarks, and classifies hand gestures to control player actions such as play, pause, forward, and rewind.

Output:
- Real-time visualization of hand landmarks and connections on the webcam feed.
- Gesture-based control of a YouTube player using hand movements.

Dependencies:
- Python libraries: cv2, mediapipe, pyautogui, time
- Ensure that the required modules are installed before running the script.

Note:
- The script assumes the presence of a single hand in the webcam feed.
- Adjustments to the gesture recognition logic can be made based on individual preferences and requirements.

Authors: Jakub Gola & Bartosz Laskowski
"""
import cv2
import mediapipe as mp
import pyautogui
import time


# Function to count fingers based on landmarks
def count_fingers(lst):
    cnt = 0

    # Calculate a threshold value based on the hand landmarks
    thresh = (lst.landmark[0].y * 100 - lst.landmark[9].y * 100) / 2

    # Check each finger by comparing the y-coordinates of specific landmarks
    if (lst.landmark[5].y * 100 - lst.landmark[8].y * 100) > thresh:
        cnt += 1

    if (lst.landmark[9].y * 100 - lst.landmark[12].y * 100) > thresh:
        cnt += 1

    if (lst.landmark[13].y * 100 - lst.landmark[16].y * 100) > thresh:
        cnt += 1

    if (lst.landmark[17].y * 100 - lst.landmark[20].y * 100) > thresh:
        cnt += 1

    # Check thumb by comparing the x-coordinates of specific landmarks
    if (lst.landmark[5].x * 100 - lst.landmark[4].x * 100) > 6:
        cnt += 1

    print(cnt)
    return cnt


# Initialize video capture, MediaPipe Hands, and other variables
cap = cv2.VideoCapture(0)
drawing = mp.solutions.drawing_utils
hands = mp.solutions.hands
hand_obj = hands.Hands(max_num_hands=1)

start_init = False
prev = -1

while True:
    end_time = time.time()

    # Read a frame from the webcam
    _, frm = cap.read()
    frm = cv2.flip(frm, 1)

    # Process the frame using MediaPipe Hands
    res = hand_obj.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    if res.multi_hand_landmarks:
        # Get landmarks for the first detected hand
        hand_keyPoints = res.multi_hand_landmarks[0]

        # Count fingers based on landmarks
        cnt = count_fingers(hand_keyPoints)

        # Gesture recognition logic
        if not (prev == cnt):
            if not (start_init):
                start_time = time.time()
                start_init = True

            elif (end_time - start_time) > 0.2:
                #fast forward 5 seconds
                if (cnt == 1):
                    pyautogui.press("right")
                #rewind 5 seconds
                elif (cnt == 2):
                    pyautogui.press("left")
                #increase volume
                elif (cnt == 3):
                    pyautogui.press("up")
                #decrease volume
                elif (cnt == 4):
                    pyautogui.press("down")
                #stop video
                elif (cnt == 5):
                    pyautogui.press("space")

                prev = cnt
                start_init = False

        # Draw landmarks and hand connections on the frame
        drawing.draw_landmarks(frm, hand_keyPoints, hands.HAND_CONNECTIONS)

    # Display the processed frame
    cv2.imshow("window", frm)

    # Exit the loop if the 'Esc' key is pressed
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.release()
        break
