import cv2
import numpy as np

def draw_grid(frame, rows, cols):
    height, width = frame.shape[:2]
    gridline_width = width // cols
    gridline_height = height // rows

    # Draw horizontal grid lines
    for i in range(1, rows):
        y = i * gridline_height
        cv2.line(frame, (0, y), (width, y), (150, 150, 150), 1)

    # Draw vertical grid lines
    for i in range(1, cols):
        x = i * gridline_width
        cv2.line(frame, (x, 0), (x, height), (150, 150, 150), 1)

# Parameters
history_weight = 0.9
min_contour_area = 1000
motion_threshold_factor = 1.5
vid_path = "../comma2k/Chunk_2/b0c9d2329ad1606b|2018-10-09--14-06-32/10/video.hevc"
my_path = '../comma2k/Chunk_1/b0c9d2329ad1606b|2018-08-17--14-55-39/7/video.hevc'

# Initialization
video_capture = cv2.VideoCapture(my_path)
rows = 20
cols = 20

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

    draw_grid(frame, rows, cols)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    if flow_history is None:
        flow_history = flow
    else:
        flow_history = history_weight * flow_history + (1 - history_weight) * flow

    magnitude, angle = cv2.cartToPolar(flow_history[..., 0], flow_history[..., 1])
    motion_threshold = np.mean(magnitude) * motion_threshold_factor
    motion_mask = magnitude > motion_threshold

    grid_cell_height = frame.shape[0] // rows
    grid_cell_width = frame.shape[1] // cols

    for i in range(rows):
        if i <= 5 or i >= 16:  # Skip rows 1-5 and 16-20
            continue
        for j in range(cols):
            if j <= 4 or j >= 16:  # Skip columns 1-4 and 16-20
                continue
            y_start = i * grid_cell_height
            x_start = j * grid_cell_width
            y_end = (i + 1) * grid_cell_height
            x_end = (j + 1) * grid_cell_width

            avg_mag = np.mean(magnitude[y_start:y_end, x_start:x_end])
            avg_angle = np.mean(angle[y_start:y_end, x_start:x_end])

            if avg_mag > motion_threshold:
                cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)

                if j in [9, 10]:  # Check columns 9 and 10
                    for col in [7, 8, 11, 12]:  # Check for movement from columns 7, 8, 11, or 12
                        # Check if there is movement from previous columns to columns 9 or 10
                        if np.mean(magnitude[y_start:y_end, col * grid_cell_width:(col + 1) * grid_cell_width]) > motion_threshold:
                            print(f"Cut-in detected from column {col} into column {j} - Mag: {avg_mag:.2f}, Angle: {avg_angle:.2f}")

    cv2.imshow("Motion Detection", frame)
    prev_gray = gray

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
