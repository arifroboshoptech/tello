
import cv2
from djitellopy import Tello

# Load the Haar cascade XML file for object detection
cascade_path = "haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(cascade_path)

# Initialize the Tello drone
tello = Tello()
tello.connect()
tello.streamon()

# Main loop for object detection
while True:
    # Read frame from the video stream
    frame = tello.get_frame_read().frame

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform object detection
    objects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw bounding boxes around the detected objects
    for (x, y, w, h) in objects:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow("Object Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
tello.streamoff()
cv2.destroyAllWindows()
tello.end()
