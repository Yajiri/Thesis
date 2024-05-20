import cv2
import numpy as np

def is_motion_towards_trapezoid(contour, trapezoid, direction):
    """Check if motion is towards the trapezoid from the specified direction."""
    rect = cv2.boundingRect(contour)
    cx, cy = rect[0] + rect[2] // 2, rect[1] + rect[3] // 2  # Center of the contour

    if direction == 'left':
        return cx < trapezoid[0][0] and cv2.pointPolygonTest(np.array(trapezoid), (cx, cy), False) >= 0
    elif direction == 'right':
        return cx > trapezoid[1][0] and cv2.pointPolygonTest(np.array(trapezoid), (cx, cy), False) >= 0
    return False

# Parameters
history_weight = 0.9
min_contour_area = 1000
motion_threshold_factor = 1.55
paths = [
    '../comma2k/Chunk_1/b0c9d2329ad1606b|2018-08-17--14-55-39/7/video.hevc', # Jen's path
    '../comma2k/Chunk_2/b0c9d2329ad1606b|2018-10-09--14-06-32/10/video.hevc', #doesnt work well
    '../comma2k/Chunk_2/b0c9d2329ad1606b|2018-09-23--12-52-06/45/video.hevc', # detects all cut-ins, no false positives 
    '../comma2k/Chunk_2/b0c9d2329ad1606b|2018-10-09--15-48-37/16/video.hevc' # works well well with SOF, but dof just shits the bed cause of shadows
]

my_path = paths[1]
# Initialization
video_capture = cv2.VideoCapture(my_path)

ret, first_frame = video_capture.read()
if not ret:
    print("Failed to read video")
    exit()
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
flow_history = None

# Define the trapezoid's coordinates using specific points
top_left = (550, 400)
top_right = (650, 400)
bottom_right = (900, 600)
bottom_left = (300, 600)
trapezoid = [top_left, top_right, bottom_right, bottom_left]

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

    motion_mask = magnitude > motion_threshold

    contours, _ = cv2.findContours(motion_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            avg_mag = np.mean(magnitude[y:y+h, x:x+w])
            avg_angle = np.mean(angle[y:y+h, x:x+w])
            
            if avg_mag > 1.2 and avg_angle > 3:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                
                # Check if the motion is towards the trapezoid
                if is_motion_towards_trapezoid(contour, trapezoid, 'left'):
                    print(f"Cut-in detected from left - Mag: {avg_mag:.2f}, Angle: {avg_angle:.2f}")
                elif is_motion_towards_trapezoid(contour, trapezoid, 'right'):
                    print(f"Cut-in detected from right - Mag: {avg_mag:.2f}, Angle: {avg_angle:.2f}")

    # Draw the trapezoid
    cv2.polylines(frame, [np.array(trapezoid)], isClosed=True, color=(0, 255, 0), thickness=2)

    cv2.imshow("Motion Detection", frame)
    prev_gray = gray

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
