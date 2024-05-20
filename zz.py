import cv2
import numpy as np

# Parameters
history_weight = 0.9
motion_threshold_factor = 1.5
paths = [
    '../comma2k/Chunk_1/b0c9d2329ad1606b|2018-08-17--14-55-39/7/video.hevc',
    '../comma2k/Chunk_2/b0c9d2329ad1606b|2018-09-23--12-52-06/45/video.hevc',
    '../comma2k/Chunk_2/b0c9d2329ad1606b|2018-10-09--15-48-37/16/video.hevc'
]

my_path = paths[1]  # Select the video path
video_capture = cv2.VideoCapture(my_path)

ret, first_frame = video_capture.read()
if not ret:
    print("Failed to read video")
    exit()
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
flow_history = None

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    if flow_history is None:
        flow_history = flow
    else:
        flow_history = history_weight * flow_history + (1 - history_weight) * flow

    magnitude, angle = cv2.cartToPolar(flow_history[..., 0], flow_history[..., 1])
    motion_threshold = np.mean(magnitude) * motion_threshold_factor

    height, width = frame.shape[:2]
    # Define ROIs
    left_roi = (100, 200, 400, 600)
    center_roi = (401, 200, 800, 600)
    right_roi = (801, 200, 1200, 600)

    # Log frame dimensions and ROI details for debugging
    print("Frame dimensions:", frame.shape)
    print("Center ROI Magnitude:", np.mean(magnitude[200:600, 401:800]))
    

    # Draw ROIs on the frame
    cv2.rectangle(frame, (left_roi[0], left_roi[1]), (left_roi[2], left_roi[3]), (255, 0, 0), 2)  # Blue rectangle for left ROI
    cv2.rectangle(frame, (right_roi[0], right_roi[1]), (right_roi[2], right_roi[3]), (255, 0, 0), 2)  # Blue rectangle for right ROI
    cv2.rectangle(frame, (center_roi[0], center_roi[1]), (center_roi[2], center_roi[3]), (255, 0, 0), 2)  # Blue rectangle for center ROI

    # Function to check motion in a region of interest
    def check_roi(roi, color):
        x_start, y_start, x_end, y_end = roi
        avg_mag = np.mean(magnitude[y_start:y_end, x_start:x_end])
        avg_angle = np.mean(angle[y_start:y_end, x_start:x_end])
        if avg_mag > motion_threshold:
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), color, 2)
        return avg_mag, avg_angle

    # Check each ROI and highlight if motion is detected
    left_mag, left_angle = check_roi(left_roi, (0, 255, 0))  # Green if motion detected
    right_mag, right_angle = check_roi(right_roi, (0, 255, 0))
    center_mag, center_angle = check_roi(center_roi, (0, 255, 0))

    # Log and display cut-in detection
    if left_mag > motion_threshold and center_mag > motion_threshold:
        print(f"Cut-in detected from left to center - Mag: {left_mag:.2f}, Angle: {left_angle:.2f}")

    if right_mag > motion_threshold and center_mag > motion_threshold:
        print(f"Cut-in detected from right to center - Mag: {right_mag:.2f}, Angle: {right_angle:.2f}")

    cv2.imshow("Motion Detection", frame)
    prev_gray = gray

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
