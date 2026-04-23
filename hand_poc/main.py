import cv2
import numpy as np
import math

DIST_WARNING = 350      
DIST_DANGER = 150       
BOX_SIZE = 150
BOX_COLOR_SAFE = (0, 255, 0)      # Green  - Safe
BOX_COLOR_WARN = (0, 255, 255)    # Yellow - Warning
BOX_COLOR_DANGER = (0, 0, 255)    # Red    - Danger

calibrated = False              
lower_skin = np.array([0, 0, 0]) 
upper_skin = np.array([179, 255, 255])

def nothing(x):
    """Dummy function required for creating OpenCV trackbars."""
    pass
def calibrate_color(frame, roi_rect):
    """
    Automatically determines HSV color bounds for the user's skin.
    Takes the median HSV values inside the calibration ROI.
    
    Parameters:
        frame (image): Current video frame
        roi_rect (tuple): (x, y, w, h) defining calibration region
    
    Returns:
        lower (array): Lower HSV threshold
        upper (array): Upper HSV threshold
    """
    x, y, w, h = roi_rect
    roi = frame[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue = np.median(hsv_roi[:, :, 0])
    sat = np.median(hsv_roi[:, :, 1])
    val = np.median(hsv_roi[:, :, 2])
    h_offset = 20
    s_offset = 50
    v_offset = 60
    lower = np.array([
        max(0, hue - h_offset),
        max(20, sat - s_offset),
        max(20, val - v_offset)
    ])
    upper = np.array([
        min(179, hue + h_offset),
        min(255, sat + s_offset),
        255
    ])
    cv2.setTrackbarPos("L - H", "Trackbars", int(lower[0]))
    cv2.setTrackbarPos("L - S", "Trackbars", int(lower[1]))
    cv2.setTrackbarPos("L - V", "Trackbars", int(lower[2]))
    cv2.setTrackbarPos("U - H", "Trackbars", int(upper[0]))
    cv2.setTrackbarPos("U - S", "Trackbars", int(upper[1]))
    cv2.setTrackbarPos("U - V", "Trackbars", int(upper[2]))
    print(f"Calibrated! Lower: {lower}, Upper: {upper}")
    return lower, upper

def main():
    global calibrated, lower_skin, upper_skin
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 300, 300)
    cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        box_x = w - BOX_SIZE - 50
        box_y = (h // 2) - (BOX_SIZE // 2)
        box_center = (box_x + BOX_SIZE // 2, box_y + BOX_SIZE // 2)
        if not calibrated:
            guide_x, guide_y = w // 2 - 50, h // 2 - 50
            guide_w, guide_h = 100, 100
            cv2.rectangle(frame, (guide_x, guide_y), (guide_x + guide_w, guide_y + guide_h), (0, 255, 0), 2)
            
            cv2.putText(frame, "STEP 1: Place hand in box", (w//2 - 150, h//2 - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "STEP 2: Press 'S' to Start", (w//2 - 150, h//2 + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.rectangle(frame, (box_x, box_y), (box_x + BOX_SIZE, box_y + BOX_SIZE), (120, 120, 120), 2)
            if cv2.waitKey(1) == ord('s'):
                lower_skin, upper_skin = calibrate_color(frame, (guide_x, guide_y, guide_w, guide_h))
                calibrated = True

        else:
            l_h = cv2.getTrackbarPos("L - H", "Trackbars")
            l_s = cv2.getTrackbarPos("L - S", "Trackbars")
            l_v = cv2.getTrackbarPos("L - V", "Trackbars")
            u_h = cv2.getTrackbarPos("U - H", "Trackbars")
            u_s = cv2.getTrackbarPos("U - S", "Trackbars")
            u_v = cv2.getTrackbarPos("U - V", "Trackbars")
            lower_skin = np.array([l_h, l_s, l_v])
            upper_skin = np.array([u_h, u_s, u_v])
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            status = "SAFE"
            status_color = BOX_COLOR_SAFE

            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 2000: 
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.drawContours(frame, [c], -1, (255, 0, 0), 2)
                        cv2.circle(frame, (cx, cy), 10, (255, 0, 0), -1)
                        cv2.line(frame, (cx, cy), box_center, (200, 200, 200), 1)
                        dist = math.sqrt((cx - box_center[0])**2 + (cy - box_center[1])**2)
                        in_box_x = box_x < cx < box_x + BOX_SIZE
                        in_box_y = box_y < cy < box_y + BOX_SIZE
                        
                        if (in_box_x and in_box_y) or dist < DIST_DANGER:
                            status = "DANGER"
                            status_color = BOX_COLOR_DANGER
                        elif dist < DIST_WARNING:
                            status = "WARNING"
                            status_color = BOX_COLOR_WARN
                        else:
                            status = "SAFE"
                            status_color = BOX_COLOR_SAFE
                        cv2.putText(frame, f"Dist: {int(dist)}px", (cx - 40, cy - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            if status == "DANGER":
                cv2.rectangle(frame, (box_x, box_y), (box_x + BOX_SIZE, box_y + BOX_SIZE),
                              status_color, -1)
                cv2.putText(frame, "DANGER!", (w//2 - 100, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            else:
                cv2.rectangle(frame, (box_x, box_y), (box_x + BOX_SIZE, box_y + BOX_SIZE),
                              status_color, 3)

            cv2.putText(frame, f"STATUS: {status}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            mask_preview = cv2.resize(mask, (160, 120))
            mask_preview = cv2.cvtColor(mask_preview, cv2.COLOR_GRAY2BGR)
            frame[h - 120:h, 0:160] = mask_preview
            cv2.putText(frame, "Mask View", (5, h - 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            if cv2.waitKey(1) == ord('r'):
                calibrated = False
        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
