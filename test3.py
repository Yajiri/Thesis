import cv2
import numpy as np

# Parameters
history_weight = 0.9
motion_threshold_factor = 1.5
accumulated_frames = 10  # Number of frames to accumulate motion for cut-in detection
vid_path = "../comma2k/Chunk_2/b0c9d2329ad1606b|2018-10-09--14-06-32/10/video.hevc"
my_path = '../comma2k/Chunk_1/b0c9d2329ad1606b|2018-08-17--14-55-39/7/video.hevc'

def detect_cut_in(prev_grays, frame, flow_histories):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Ensure prev_grays is a list
    if isinstance(prev_grays, np.ndarray):
        prev_grays = [prev_grays]

    prev_gray = prev_grays[-1] if prev_grays else gray
    
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Initialize or update flow histories
    if len(flow_histories) < accumulated_frames:
        flow_histories.append(flow)
    else:
        flow_histories.pop(0)
        flow_histories.append(flow)

    # Accumulate motion across frames
    magnitude_accumulator = np.zeros_like(flow[..., 0])
    for flow_history in flow_histories:
        magnitude, _ = cv2.cartToPolar(flow_history[..., 0], flow_history[..., 1])
        magnitude_accumulator += magnitude

    # Calculate mean magnitude across accumulated frames
    mean_magnitude = magnitude_accumulator / accumulated_frames
    motion_threshold = np.mean(mean_magnitude) * motion_threshold_factor

    # Detect cut-in if there is significant motion from the sides towards the middle
    height, width = frame.shape[:2]
    left_region = (0, 0, width // 3, height)  # Left third of the frame
    right_region = (2 * width // 3, 0, width // 3, height)  # Right third of the frame

    left_magnitude = np.mean(mean_magnitude[:, :width // 3])
    right_magnitude = np.mean(mean_magnitude[:, 2 * width // 3:])

    if left_magnitude > motion_threshold and right_magnitude > motion_threshold:
        print("Cut-in detected!")
        # Annotate the frame with cut-in detection
        cv2.putText(frame, "Cut-in Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return gray, flow_histories

# Initialization
video_capture = cv2.VideoCapture(vid_path)
prev_grays = None
flow_histories = []

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    prev_grays, flow_histories = detect_cut_in(prev_grays, frame, flow_histories)

    # Display the frame
    cv2.imshow('Video with Cut-in Detection', frame)

    # Check for exit key
    if cv2.waitKey(1) == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
