import cv2
import numpy as np

# --- CONFIGURATION ---
VIDEO_BROADCAST_PATH = "src/data/broadcast.mp4"
VIDEO_TACTICAM_PATH = "src/data/tacticam.mp4"
# ---------------------

# Global variables to store the points and video captures/frames
points_broadcast = []
points_tacticam = []

cap_b = None
cap_t = None
current_frame_b = None
current_frame_t = None
frame_b_display = None # Copy of frame_b for drawing
frame_t_display = None # Copy of frame_t for drawing

def mouse_callback_broadcast(event, x, y, flags, param):
    """Mouse callback function for the broadcast window."""
    global frame_b_display, points_broadcast
    if event == cv2.EVENT_LBUTTONDOWN:
        points_broadcast.append((x, y))
        # Redraw circle and text on the display copy
        cv2.circle(frame_b_display, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(frame_b_display, f"{len(points_broadcast)}", (x+10, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Broadcast View - Click Points", frame_b_display)
        print(f"Broadcast Point {len(points_broadcast)} added: ({x}, {y})")

def mouse_callback_tacticam(event, x, y, flags, param):
    """Mouse callback function for the tactical window."""
    global frame_t_display, points_tacticam
    if event == cv2.EVENT_LBUTTONDOWN:
        points_tacticam.append((x, y))
        # Redraw circle and text on the display copy
        cv2.circle(frame_t_display, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(frame_t_display, f"{len(points_tacticam)}", (x+10, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Tacticam View - Click Points", frame_t_display)
        print(f"Tacticam Point {len(points_tacticam)} added: ({x}, {y})")

def update_frames(frame_num_b, frame_num_t):
    """Reads and displays frames at specified numbers for both videos."""
    global cap_b, cap_t, current_frame_b, current_frame_t, frame_b_display, frame_t_display

    if cap_b is None or cap_t is None:
        print("Error: Video captures not initialized.")
        return False

    # Broadcast video
    cap_b.set(cv2.CAP_PROP_POS_FRAMES, frame_num_b)
    success_b, temp_frame_b = cap_b.read()
    if success_b:
        current_frame_b = temp_frame_b
        frame_b_display = current_frame_b.copy()
        # Redraw existing points on the new frame copy
        for i, (x, y) in enumerate(points_broadcast):
            cv2.circle(frame_b_display, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(frame_b_display, f"{i+1}", (x+10, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Broadcast View - Click Points", frame_b_display)
    else:
        print(f"Warning: Could not read frame {frame_num_b} from broadcast video.")
        return False

    # Tacticam video
    cap_t.set(cv2.CAP_PROP_POS_FRAMES, frame_num_t)
    success_t, temp_frame_t = cap_t.read()
    if success_t:
        current_frame_t = temp_frame_t
        frame_t_display = current_frame_t.copy()
        # Redraw existing points on the new frame copy
        for i, (x, y) in enumerate(points_tacticam):
            cv2.circle(frame_t_display, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(frame_t_display, f"{i+1}", (x+10, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Tacticam View - Click Points", frame_t_display)
    else:
        print(f"Warning: Could not read frame {frame_num_t} from tacticam video.")
        return False
    
    return success_b and success_t


if __name__ == "__main__":
    print("--- Homography Point Selection Tool with Frame Navigation ---")
    print("Instructions:")
    print("1. Two windows will open. Use 'a' to go back one frame, 'd' to go forward one frame.")
    print("2. Navigate to a frame in *both* videos where common static points are clearly visible.")
    print("3. Identify a static point on the field (e.g., a corner of the penalty box).")
    print("4. IMPORTANT: Click on that exact point FIRST in the 'Tacticam' window, and THEN on the SAME point in the 'Broadcast' window.")
    print("5. Repeat this process for at least 4 points. The more points, the better the result.")
    print("6. The order of clicks MUST be maintained: Tacticam Point 1 -> Broadcast Point 1, Tacticam Point 2 -> Broadcast Point 2, ...")
    print("7. You can clear all points and restart point selection on the current frame by pressing 'c'.")
    print("8. When you are finished, press the 'q' key on either window to close and print the results.")
    print("-" * 70)

    cap_b = cv2.VideoCapture(VIDEO_BROADCAST_PATH)
    cap_t = cv2.VideoCapture(VIDEO_TACTICAM_PATH)

    if not cap_b.isOpened():
        print(f"Error: Could not open broadcast video at {VIDEO_BROADCAST_PATH}")
        exit()
    if not cap_t.isOpened():
        print(f"Error: Could not open tacticam video at {VIDEO_TACTICAM_PATH}")
        exit()

    frame_count_b = int(cap_b.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count_t = int(cap_t.get(cv2.CAP_PROP_FRAME_COUNT))

    current_frame_num_b = 0
    current_frame_num_t = 0

    cv2.namedWindow("Broadcast View - Click Points")
    cv2.setMouseCallback("Broadcast View - Click Points", mouse_callback_broadcast)

    cv2.namedWindow("Tacticam View - Click Points")
    cv2.setMouseCallback("Tacticam View - Click Points", mouse_callback_tacticam)

    # Initial frame display
    if not update_frames(current_frame_num_b, current_frame_num_t):
        print("Failed to load initial frames. Exiting.")
        exit()

    while True:
        key = cv2.waitKey(1) & 0xFF # Wait for a key press

        if key == ord('q'):
            break
        elif key == ord('d'): # Move forward one frame
            current_frame_num_b = min(current_frame_num_b + 1, frame_count_b - 1)
            current_frame_num_t = min(current_frame_num_t + 1, frame_count_t - 1)
            print(f"Moving to frame B:{current_frame_num_b}, T:{current_frame_num_t}")
            update_frames(current_frame_num_b, current_frame_num_t)
        elif key == ord('a'): # Move backward one frame
            current_frame_num_b = max(0, current_frame_num_b - 1)
            current_frame_num_t = max(0, current_frame_num_t - 1)
            print(f"Moving to frame B:{current_frame_num_b}, T:{current_frame_num_t}")
            update_frames(current_frame_num_b, current_frame_num_t)
        elif key == ord('c'): # Clear all selected points
            points_broadcast.clear()
            points_tacticam.clear()
            print("Cleared all selected points. Restarting selection for current frame.")
            # Redraw current frames to remove old circles/text
            update_frames(current_frame_num_b, current_frame_num_t)


    cv2.destroyAllWindows()
    cap_b.release()
    cap_t.release()

    # --- Print the final arrays ---
    if len(points_tacticam) != len(points_broadcast):
        print("\n\n!! WARNING: You have selected a different number of points for each view.")
        print("The number of points must be equal. Please run the script again and be careful with your clicks.")
    elif len(points_tacticam) < 4:
        print("\n\n!! WARNING: You need at least 4 points to calculate homography.")
    else:
        print("\n\n--- Copy and paste the following arrays into your main script ---\n")
        
        # Format the points into a numpy array string for easy copy-pasting
        pts_tacticam_str = np.array2string(np.array(points_tacticam), 
                                            separator=', ', 
                                            formatter={'float_kind':lambda x: "%.0f" % x})
        print(f"pts_tacticam = np.array({pts_tacticam_str.replace('[', '[').replace(']', ']')}, dtype=np.float32)")

        pts_broadcast_str = np.array2string(np.array(points_broadcast), 
                                            separator=', ', 
                                            formatter={'float_kind':lambda x: "%.0f" % x})
        print(f"pts_broadcast = np.array({pts_broadcast_str.replace('[', '[').replace(']', ']')}, dtype=np.float32)")