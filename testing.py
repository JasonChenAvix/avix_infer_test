import cv2
from object_detection_util import KalmanFilter ,ReIDTrack 
import time

# Open the video file
video_path = 'classroom.mp4'
cap = cv2.VideoCapture(video_path)
model = ReIDTrack()
while True:
    # Check if the video has reached its end
    if not cap.isOpened():
        # Reopen the video file
        cap = cv2.VideoCapture(video_path)

    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform object tracking
    tracked_objects = model.track(frame)

    time.sleep(3)
# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
