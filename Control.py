import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import speech_recognition as sr
import threading
import time
#import google.generativeai as genai

#genai.configure(api_key="AIzaSyBKJoz0rbdnWkG9sKnFj32x5f5oZhA2I-o")
#model = genai.GenerativeModel("gemini-1.5-flash")

#? Global Vars
labels = {0: 'Forward', 1:'Backward', 2:'Up', 3:'Down', 4:'Left', 5:'Right'}
model = tf.keras.models.load_model('landmark-model.keras')

pred_left, pred_right = '', ''
temp_left, temp_right = '', ''
count_left, count_right = 0, 0

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)
draw = mp.solutions.drawing_utils

# open the camera
vid = cv2.VideoCapture(0)

def recognize_speech():
    recognizer = sr.Recognizer()
    while True:  # Continuous speech recognition
        with sr.Microphone() as source:
            print("Speech Recognition: Listening...")
            recognizer.adjust_for_ambient_noise(source)  # Reduce noise interference
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                command = recognizer.recognize_google(audio)
                print(f"Speech Recognized: {command}")
                #response = model.generate_content(f"This is a trivia question: {command}, answer in yes or no only. even if it's not a trivia, just make up things, but answer in a yes or no only.")
                #print(response.text[-1])
            except sr.UnknownValueError:
                print("Speech Recognition: Could not understand audio.")
            except sr.RequestError as e:
                print(f"Speech Recognition: Request error {e}")
            except sr.WaitTimeoutError:
                print("Speech Recognition: No speech detected within the timeout.")
            time.sleep(0.1)

# Start Speech Recognition Thread
speech_thread = threading.Thread(target=recognize_speech, daemon=True)
speech_thread.start()

while True:
    ret, frame = vid.read()
    if not ret or frame is None:
        continue  # Skip the iteration if the frame isn't captured properly

    frame = cv2.flip(frame, 1)
    y, x, n = frame.shape

    # Preprocessing the image
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb_img)

    landmarks = []
    hands_num = []

    if res.multi_hand_landmarks:
        # Drawing landmarks on the frame
        for id, handslms in enumerate(res.multi_hand_landmarks):
            draw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            hands_num.append(res.multi_handedness[id].classification[0].label)

            # storing landmarks to feed the model
            pos = []
            for lm in handslms.landmark:
                pos.append(lm.x * x)
                pos.append(lm.y * y)

            landmarks.append(pos)

        # Asking the model to predict the gesture
        landmarks = np.array(landmarks)
        landmarks = landmarks.reshape(-1, landmarks.shape[1])
        prob = model.predict(landmarks)
        upper = 50

        for i, hand in zip(prob, hands_num):
            className = np.argmax(i)

            if i[className] > 0.9:
                gesture = labels[className]

                if hand == 'Left':
                    if gesture == temp_left:
                        count_left += 1
                    else:
                        temp_left = gesture
                        count_left = 0

                    if count_left == 3:
                        pred_left = temp_left
                        count_left = 0

                    className = hand + ': ' + pred_left
                    
                
                elif hand == 'Right':
                    if gesture == temp_right:
                        count_right += 1
                    else:
                        temp_right = gesture
                        count_right = 0

                    if count_right == 3:
                        pred_right = temp_right
                        count_right = 0

                    className = hand + ': ' + pred_right
            else:
                className = ''

            cv2.putText(frame, className, (300, upper), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)
            upper += 50

    # Show the output
    cv2.imshow('Video Output', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()