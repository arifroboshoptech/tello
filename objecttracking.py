import cv2
import numpy as np
from djitellopy import Tello

# Load the Haar cascade XML file for face detection
cascade_path = "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cascade_path)

# Initialize the Tello drone
tello = Tello()
tello.connect()
tello.streamon()

# Variables for tracking movement
prev_x, prev_y = 0, 0

# Main loop for face detection and tracking
while True:
    # Read frame from the video stream
    frame = tello.get_frame_read().frame

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Track movement
    x, y, w, h = 0, 0, 0, 0
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Calculate the centroid of the detected face
        center_x = x + w // 2
        center_y = y + h // 2

        # Calculate the movement based on the centroid
        move_x = center_x - prev_x
        move_y = center_y - prev_y

        # Track movement by sending commands to the Tello drone
        if abs(move_x) > 10 or abs(move_y) > 10:
            if move_x > 20:
                tello.move_right(20)
            elif move_x < -20:
                tello.move_left(20)
            if move_y > 20:
                tello.move_down(20)
            elif move_y < -20:
                tello.move_up(20)

        # Update previous centroid coordinates
        prev_x, prev_y = center_x, center_y

    # Display the frame with face detection and movement tracking
    cv2.imshow("Face Detection and Tracking", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
tello.streamoff()
cv2.destroyAllWindows()
tello.end()
