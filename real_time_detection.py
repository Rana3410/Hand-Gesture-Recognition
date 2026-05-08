import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# Set the model path (make sure this points to where you saved your model)
MODEL_PATH = 'Training/hgr_model.h5'

# 1. Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model '{MODEL_PATH}' not found.")
    print("Please go to your Jupyter notebook, train the model, and save it using: model.save('hgr_model.h5')")
    exit()

# Load your saved model
print("Loading model... This might take a moment.")
model = load_model(MODEL_PATH)

# 2. Define the gesture labels (Must match the exact order from your training data)
labels = ['01_palm', '02_l', '03_fist', '04_fist_moved', '05_thumb', 
          '06_index', '07_ok', '08_palm_moved', '09_c', '10_down']

# 3. Setup Background Subtraction (to extract the hand cleanly)
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)

# 4. Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define the Region of Interest (ROI) coordinates where the user should place their hand
roi_top, roi_bottom, roi_right, roi_left = 100, 350, 300, 550

print("Starting webcam feed... Press 'q' to exit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Flip the frame horizontally to act like a mirror
    frame = cv2.flip(frame, 1)
    
    # Draw the ROI rectangle on the main frame
    cv2.rectangle(frame, (roi_right, roi_top), (roi_left, roi_bottom), (0, 255, 0), 2)
    cv2.putText(frame, "Place Hand Here", (roi_right, roi_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Extract the ROI (the area inside the green box)
    roi = frame[roi_top:roi_bottom, roi_right:roi_left]
    
    # Apply Background Subtraction to the ROI
    fg_mask = bg_subtractor.apply(roi)
    
    # Clean up the mask (remove noise)
    kernel = np.ones((3, 3), np.uint8)
    fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
    
    # Apply the mask to the ROI to extract only the hand (black out the background)
    extracted_hand = cv2.bitwise_and(roi, roi, mask=fg_mask)
    
    # -- PREPARE IMAGE FOR MODEL PREDICTION --
    # Resize to match the input size of your model (150x150 for your custom CNN)
    img = cv2.resize(extracted_hand, (150, 150)) 
    
    # Convert from BGR (OpenCV default) to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Rescale pixel values (just like you did with ImageDataGenerator)
    img = img / 255.0
    
    # Expand dimensions to create a batch of 1
    img = np.expand_dims(img, axis=0)
    
    # 5. Predict the gesture
    prediction = model.predict(img, verbose=0)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)
    
    # If the model is more than 60% confident, show the label
    if confidence > 0.6: 
        gesture_name = labels[class_index]
        text = f"{gesture_name} ({confidence*100:.1f}%)"
        cv2.putText(frame, text, (roi_right, roi_bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    
    # Display the main camera feed
    cv2.imshow('Real-time Hand Gesture Recognition', frame)
    
    # Display the extracted hand window so you can see how the background subtraction is working
    cv2.imshow('Hand Extraction (Background Subtracted)', extracted_hand)
    
    # Press 'q' on your keyboard to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
