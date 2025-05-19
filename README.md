🖐️ Hand Gesture & Speech-Controlled Interface using TensorFlow + Mediapipe
This project enables real-time control and command recognition using hand gestures and speech input, powered by MediaPipe, TensorFlow, and speech recognition APIs. It can be used in interactive systems such as robotic control, games, or accessibility applications.

📁 Project Structure
bash
Copy
Edit
├── Control.py         # Main program for real-time gesture + voice recognition
├── dataset.ipynb      # Notebook to collect and visualize landmark data
├── model.ipynb        # Notebook to train a classification model on hand gestures
├── landmark-model.keras  # Trained model file (loaded in Control.py)
🚀 Features
✋ Real-time hand gesture recognition using MediaPipe Hands

🗣️ Voice command recognition using Google's speech recognition API

🧠 Deep learning model predicts gestures from 21 hand landmarks (trained in Keras)

🧪 Intelligent noise filtering: gesture is only accepted after consistent detection

🔁 Runs continuously with live webcam and microphone input

⚙️ Setup Instructions
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/gesture-speech-control.git
cd gesture-speech-control
Install dependencies:
Make sure you have Python 3.7+ and install the following:

bash
Copy
Edit
pip install opencv-python mediapipe numpy tensorflow speechrecognition
Collect data and train model (optional):

Run dataset.ipynb to capture labeled gesture data.

Train the model using model.ipynb.

Run the controller:

bash
Copy
Edit
python Control.py
✋ Recognized Gestures
The model can predict the following hand gestures:

Label	Meaning
Forward	Move forward
Backward	Move backward
Up	Move up
Down	Move down
Left	Move left
Right	Move right

Each hand (left/right) is recognized and labeled independently.

🎤 Speech Recognition
In parallel with gesture detection, the system listens for speech using your microphone and transcribes it live using Google's free speech API.

🔒 Notes
The model file landmark-model.keras must be present in the same directory as Control.py.

Internet connection is required for speech recognition (since it uses Google’s API).

genai support (e.g., Gemini or Bard) is present in comments if you want to expand it into a trivia assistant.
