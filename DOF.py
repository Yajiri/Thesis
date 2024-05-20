import cv2
import numpy as np

def is_motion_towards_trapezoid(contour, trapezoid, direction, flow, magnitude, angle):
    rect = cv2.boundingRect(contour)
    cx, cy = rect[0] + rect[2] // 2, rect[1] + rect[3] // 2  # Center of the contour
    contour_poly = cv2.approxPolyDP(contour, 3, True)

    if direction == 'left':
        if cx < trapezoid[0][0]:
            for point in contour_poly:
                px, py = point[0]  # Convert point to tuple
                if cv2.pointPolygonTest(np.array(trapezoid), (int(px), int(py)), False) >= 0:
                    if np.mean(flow[int(py), int(px), 0]) > 0 and np.mean(magnitude[int(py), int(px)]) > 1.2 and 0 < np.mean(angle[int(py), int(px)]) < np.pi:
                        return True
    elif direction == 'right':
        if cx > trapezoid[1][0]:
            for point in contour_poly:
                px, py = point[0]  # Convert point to tuple
                if cv2.pointPolygonTest(np.array(trapezoid), (int(px), int(py)), False) >= 0:
                    if np.mean(flow[int(py), int(px), 0]) < 0 and np.mean(magnitude[int(py), int(px)]) > 1.2 and np.pi < np.mean(angle[int(py), int(px)]) < 2 * np.pi:
                        return True
    return False

# Parameters
history_weight = 0.9
min_contour_area = 1000
max_contour_area = 20000
aspect_ratio_min = 0.5
aspect_ratio_max = 4.0
motion_threshold_factor = 1.5
blur_ksize = 5  # Kernel size for blurring

paths = [
    '../comma2k/Chunk_1/b0c9d2329ad1606b|2018-08-17--14-55-39/7/video.hevc',
    '../comma2k/Chunk_2/b0c9d2329ad1606b|2018-10-09--14-06-32/10/video.hevc',
    '../comma2k/Chunk_2/b0c9d2329ad1606b|2018-09-23--12-52-06/45/video.hevc',
    '../comma2k/Chunk_2/b0c9d2329ad1606b|2018-10-09--15-48-37/16/video.hevc'
]

my_path = paths[2]
# Initialization
video_capture = cv2.VideoCapture(my_path)

ret, first_frame = video_capture.read()
if not ret:
    print("Failed to read video")
    exit()
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
flow_history = None

# Define the trapezoid's coordinates using specific points
top_left = (585, 385)
top_right = (615, 385)
bottom_right = (850, 600)
bottom_left = (350, 600)
trapezoid = [top_left, top_right, bottom_right, bottom_left]

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, blur_ksize)  # Apply median blur to reduce noise

    fgmask = fgbg.apply(gray)
    bg_removed = cv2.bitwise_and(gray, gray, mask=fgmask)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, bg_removed, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    if flow_history is None:
        flow_history = flow
    else:
        flow_history = history_weight * flow_history + (1 - history_weight) * flow

    magnitude, angle = cv2.cartToPolar(flow_history[..., 0], flow_history[..., 1])
    motion_threshold = np.mean(magnitude) * motion_threshold_factor

    motion_mask = magnitude > motion_threshold

    contours, _ = cv2.findContours(motion_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if min_contour_area < cv2.contourArea(contour) < max_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            if aspect_ratio_min < aspect_ratio < aspect_ratio_max:
                avg_mag = np.mean(magnitude[y:y+h, x:x+w])
                avg_angle = np.mean(angle[y:y+h, x:x+w])

                if avg_mag > 1.2 and avg_angle > 3:
                    cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)

                    if is_motion_towards_trapezoid(contour, trapezoid, 'left', flow, magnitude, angle):
                        print(f"Cut-in detected from left - Mag: {avg_mag:.2f}, Angle: {avg_angle:.2f}")
                    elif is_motion_towards_trapezoid(contour, trapezoid, 'right', flow, magnitude, angle):
                        print(f"Cut-in detected from right - Mag: {avg_mag:.2f}, Angle: {avg_angle:.2f}")

    cv2.polylines(frame, [np.array(trapezoid)], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.imshow("Motion Detection", frame)
    prev_gray = gray

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
