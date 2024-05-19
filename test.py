import cv2
import numpy as np

# Parameters
history_weight = 0.9
min_contour_area = 1000
aspect_ratio_min = 0.5
aspect_ratio_max = 4.0
motion_threshold_factor = 1.5
vid_path = "../comma2k/Chunk_2/b0c9d2329ad1606b|2018-10-09--14-06-32/10/video.hevc"
my_path = '../comma2k/Chunk_1/b0c9d2329ad1606b|2018-08-17--14-55-39/7/video.hevc'

def detect_cut_in(prev_gray, frame, flow_history):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Convert frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Define lower and upper bounds for cyan color in HSV
    lower_cyan = np.array([80, 100, 100])
    upper_cyan = np.array([100, 255, 255])
    # Create mask for cyan color
    cyan_mask = cv2.inRange(hsv_frame, lower_cyan, upper_cyan)
    # Set cyan-colored pixels to black in the gray frame (ignoring them in optical flow calculation)
    gray[cyan_mask > 0] = 0
    
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    if flow_history is None:
        flow_history = flow
    else:
        flow_history = history_weight * flow_history + (1 - history_weight) * flow

    magnitude, angle = cv2.cartToPolar(flow_history[..., 0], flow_history[..., 1])
    motion_threshold = np.mean(magnitude) * motion_threshold_factor
    motion_mask = magnitude > motion_threshold

    # Initialize a mask for visualization
    vis_mask = np.zeros_like(frame)
    vis_mask[..., 1] = 255  # Set saturation to maximum

    contours, _ = cv2.findContours(motion_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area < min_contour_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if aspect_ratio_min < aspect_ratio < aspect_ratio_max:
            # Calculate highest speed and angle within the contour
            local_mag = magnitude[y:y+h, x:x+w]
            local_angle = angle[y:y+h, x:x+w]
            speed = np.max(local_mag)
            mean_angle = np.degrees(np.mean(local_angle))

            # Annotate the frame
            aspect_ratio_text = f"Aspect Ratio: {aspect_ratio:.2f}"
            area_text = f"Area: {contour_area:.2f} px"
            speed_text = f"Speed: {speed:.2f} px/frame"
            angle_text = f"Angle: {mean_angle:.0f}°"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, aspect_ratio_text, (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, area_text, (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, speed_text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, angle_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Update visualization mask
            vis_mask[y:y+h, x:x+w, 0] = local_angle * 180 / np.pi / 2
            vis_mask[y:y+h, x:x+w, 2] = cv2.normalize(local_mag, None, 0, 255, cv2.NORM_MINMAX)

    # Convert HSV to RGB (BGR) color representation
    rgb = cv2.cvtColor(vis_mask, cv2.COLOR_HSV2BGR)
    combined = cv2.addWeighted(frame, 1, rgb, 0.6, 0)

    cv2.imshow('Motion Detection with Flow Visualization', combined)
    return gray, flow_history

# Initialization
video_capture = cv2.VideoCapture(my_path)
ret, first_frame = video_capture.read()
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
flow_history = None

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    prev_gray, flow_history = detect_cut_in(prev_gray, frame, flow_history)

    if cv2.waitKey(1) == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
