import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 1. Load your trained model
model = load_model('mask_model.h5')

# 2. Load a pre-trained face detector (built into OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Labels for your model output
labels = {0: "Mask", 1: "No Mask", 2: "Partial Mask"}
colors = {0: (0, 255, 0), 1: (0, 0, 255), 2: (0, 255, 255)} # BGR format

cap = cv2.VideoCapture(0)

print("Starting video feed... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    for (x, y, w, h) in faces:
        # Crop the face and resize to 64x64
        face_roi = gray_frame[y:y+h, x:x+w]
        resized_face = cv2.resize(face_roi, (64, 64))
        
        # Normalize and reshape for the CNN (Batch, Width, Height, Channel)
        normalized_face = resized_face / 255.0
        reshaped_face = np.reshape(normalized_face, (1, 64, 64, 1))

        # Predict
        prediction = model.predict(reshaped_face, verbose=0)
        result_idx = np.argmax(prediction)
        label_text = labels[result_idx]
        confidence = np.max(prediction) * 100

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), colors[result_idx], 2)
        cv2.putText(frame, f"{label_text} ({confidence:.1f}%)", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[result_idx], 2)

    cv2.imshow('Face Mask Detector - Live Demo', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()