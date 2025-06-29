from ultralytics import YOLO
import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2
import collections
import pickle
import os
import logging

# 
# 
#
#

# --- Configuration (Keep your paths and initial points)
BROADCAST_VIDEO_PATH = 'src/data/broadcast.mp4'
TACTICAM_VIDEO_PATH = 'src/data/tacticam.mp4'
OUTPUT_VIDEO_PATH = 'combined_output_optimized.mp4'
MODEL_PATH = 'src/model/best.pt'


# --- HOMOGRAPHY CONFIG

# pts_tacticam = np.array([[464, 255],
#  [ 64, 691],
#  [229, 514],
#  [367, 362],
#  [ 42, 231],
#  [169, 138],
#  [ 70, 332],
#  [135, 426]], dtype=np.float32)

# pts_broadcast = np.array([[1731,  571],
#  [ 704,  927],
#  [1125,  774],
#  [1485,  651],
#  [ 988,  524],
#  [1256,  463],
#  [ 974,  603],
#  [1019,  686]], dtype=np.float32)


# -----------------------------------------------
# Calcuated using selector_points.py
# by clicking same points on both videos (maually)
# 
# -------------------------------------------------

pts_tacticam = np.array([[517, 361],
 [566, 422],
 [385, 504],
 [230, 670],
 [ 40, 656],
 [300, 417],
 [481, 251],
 [352, 242],
 [243, 326],
 [ 24, 506],
 [ 37, 362],
 [ 27, 298],
 [102, 316],
 [218, 234],
 [336, 146],
 [459, 153]], dtype=np.float32)
pts_broadcast = np.array([[1788,  669],
 [1862,  728],
 [1412,  789],
 [ 986,  937],
 [ 645,  895],
 [1303,  696],
 [1773,  570],
 [1508,  552],
 [1260,  614],
 [ 734,  750],
 [ 868,  624],
 [ 867,  495],
 [1013,  593],
 [1276,  535],
 [1549,  473],
 [1776,  486]], dtype=np.float32)


# --- MATCHING CONFIG
# --- OPTIMIZATION -  Normalized weights for a more stable cost function
SPATIAL_COST_WEIGHT = 0.6
VISUAL_COST_WEIGHT = 0.2
MOTION_COST_WEIGHT = 0.2 # New weight for motion consistency
MATCH_CONFIDENCE_THRESHOLD = 0.7


def get_player_center(bbox):
    return ((bbox[0] + bbox[2]) / 2, bbox[3])

def get_color_histogram(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    player_roi = frame[y1:y2, x1:x2]
    if player_roi.size == 0:
        return np.zeros(180)
    hsv_crop = cv2.cvtColor(player_roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_crop, np.array([0, 70, 50]), np.array([180, 255, 255]))
    hist = cv2.calcHist([hsv_crop], [0], mask, [180], [0, 180])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten()

def update_homography():
    homography_matrix, _ = cv2.findHomography(pts_tacticam, pts_broadcast)
    if homography_matrix is None:
        logging.error("Failed to get homography.")
        return np.identity(3)
    return homography_matrix

# --- Core Video Processing (with motion history)
def process_video(VIDEO_PATH, model):
    cache_file = os.path.splitext(VIDEO_PATH)[0] + '_tracks_v2.pkl'
    if os.path.exists(cache_file):
        logging.info(f"Loading cached tracks from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    logging.info(f"Processing video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    all_tracks = collections.defaultdict(dict)

    frame_num = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_num += 1
        results = model.track(frame, persist=True, tracker='botsort.yaml', verbose=False)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, track_ids):
                # Store position history for velocity calculation
                if track_id not in all_tracks[frame_num]:
                    all_tracks[frame_num][track_id] = { "positions": collections.deque(maxlen=5) }

                all_tracks[frame_num][track_id]["bbox"] = box
                all_tracks[frame_num][track_id]["color_hist"] = get_color_histogram(frame, box)
                all_tracks[frame_num][track_id]["positions"].append(get_player_center(box))

    cap.release()
    with open(cache_file, 'wb') as f:
        pickle.dump(dict(all_tracks), f)
    logging.info(f"Finished processing video: {VIDEO_PATH}")
    return dict(all_tracks)

# --- ID Matching Class
class IDMatcher:
    def __init__(self):
        # {t_id: {b_id: score, ...}, ...}
        self.match_scores = collections.defaultdict(lambda: collections.defaultdict(float))
        # {t_id: confirmed_b_id}
        self.confirmed_matches = {}

    def update(self, cost_matrix, t_ids, b_ids):
        if not t_ids or not b_ids:
            return

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Decay scores for all potential matches to "forget" old, weak evidence
        for t_id in self.match_scores:
            for b_id in list(self.match_scores[t_id]):
                self.match_scores[t_id][b_id] *= 0.9 # Decay factor

        # Update scores with new matches
        for b_idx, t_idx in zip(row_ind, col_ind):
            cost = cost_matrix[b_idx, t_idx]
            # Convert cost to score (lower cost = higher score)
            score = max(0, 1 - cost)

            if score > 0.1: # Only reasonably good matches
                b_id = b_ids[b_idx]
                t_id = t_ids[t_idx]
                self.match_scores[t_id][b_id] += score

        # Promote scores to confirmed matches if they are confident enough
        for t_id, scores in self.match_scores.items():
            if not scores: continue
            
            best_b_id = max(scores, key=scores.get)
            best_score = scores[best_b_id]
            
            # Check if this match is confident and not already claimed
            if best_score > MATCH_CONFIDENCE_THRESHOLD and best_b_id not in self.confirmed_matches.values():
                self.confirmed_matches[t_id] = best_b_id

    def get_id(self, t_id):
        return self.confirmed_matches.get(t_id)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    model = YOLO(MODEL_PATH)

    tacticam_tracks = process_video(TACTICAM_VIDEO_PATH, model)
    broadcast_tracks = process_video(BROADCAST_VIDEO_PATH, model)

    cap_broadcast = cv2.VideoCapture(BROADCAST_VIDEO_PATH)
    cap_tacticam = cv2.VideoCapture(TACTICAM_VIDEO_PATH)
    

    w_broadcast = int(cap_broadcast.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_broadcast = int(cap_broadcast.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_tacticam = int(cap_tacticam.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_tacticam = int(cap_tacticam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap_broadcast.get(cv2.CAP_PROP_FPS))
    output_width = w_broadcast + w_tacticam
    output_height = max(h_broadcast, h_tacticam)
    output_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (output_width, output_height))


    id_matcher = IDMatcher()
    frame_idx = 0
    max_spatial_dist = np.linalg.norm([w_broadcast, h_broadcast])

    while cap_broadcast.isOpened() and cap_tacticam.isOpened():
        frame_idx += 1
        success_b, frame_b = cap_broadcast.read()
        success_t, frame_t = cap_tacticam.read()
        if not success_b or not success_t: break

        homography_matrix = update_homography()

        t_players = tacticam_tracks.get(frame_idx, {})
        b_players = broadcast_tracks.get(frame_idx, {})

        if b_players and t_players:
            b_ids, b_data = zip(*b_players.items())
            t_ids, t_data = zip(*t_players.items())
            num_b, num_t = len(b_ids), len(t_ids)
            cost_matrix = np.full((num_b, num_t), 1.0) # Cost now normalized [0, 1]

            for i in range(num_t):
                # Project tactical player position
                t_pos = get_player_center(t_data[i]['bbox'])
                t_pos_h = np.dot(homography_matrix, [t_pos[0], t_pos[1], 1])
                t_pos_proj = (t_pos_h[:2] / t_pos_h[2])

                # Calculate tactical player velocity
                t_vel = np.array([0,0])
                if len(t_data[i]['positions']) > 1:
                    t_vel = np.array(t_data[i]['positions'][-1]) - np.array(t_data[i]['positions'][-2])
                
                # Project velocity vector (simplified: project end point)
                t_vel_end_proj_h = np.dot(homography_matrix, [t_pos[0] + t_vel[0], t_pos[1] + t_vel[1], 1])
                t_vel_end_proj = t_vel_end_proj_h[:2] / t_vel_end_proj_h[2]
                t_vel_proj = t_vel_end_proj - t_pos_proj

                for j in range(num_b):
                    b_pos = get_player_center(b_data[j]['bbox'])
                    
                    # ----------------------------------------
                    # 2. NORMALIZED & ENHANCED COST FUNCTION
                    # ----------------------------------------
                    # Spatial Cost (Normalized)
                    spatial_dist = np.linalg.norm(t_pos_proj - b_pos)
                    spatial_cost = min(1.0, spatial_dist / max_spatial_dist)

                    # Visual Cost
                    visual_cost = cv2.compareHist(t_data[i]['color_hist'], b_data[j]['color_hist'], cv2.HISTCMP_BHATTACHARYYA)

                    # Motion Cost (Normalized)
                    b_vel = np.array([0,0])
                    if len(b_data[j]['positions']) > 1:
                        b_vel = np.array(b_data[j]['positions'][-1]) - np.array(b_data[j]['positions'][-2])
                    
                    # Compare velocity vectors (cosine similarity would be even better)
                    motion_diff = np.linalg.norm(t_vel_proj - b_vel)
                    motion_cost = min(1.0, motion_diff / 50) # Normalize by a reasonable max pixel velocity

                    # Total Weighted Cost
                    total_cost = (SPATIAL_COST_WEIGHT * spatial_cost +
                                  VISUAL_COST_WEIGHT * visual_cost +
                                  MOTION_COST_WEIGHT * motion_cost)
                    cost_matrix[j, i] = total_cost
            
            # 3. ID MAPPING
            id_matcher.update(cost_matrix, t_ids, b_ids)

        # 4. VISUALIZATION
        output_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        output_frame[:h_tacticam, :w_tacticam] = frame_t # Corrected placement
        output_frame[:h_broadcast, w_tacticam:] = frame_b # Corrected placement

        # Drawing boxes with confirmed IDs in tacticam view
        for t_id, t_info in t_players.items():
            x1, y1, x2, y2 = map(int, t_info['bbox'])
            confirmed_id = id_matcher.get_id(t_id)
            display_text = f"B_{confirmed_id}" if confirmed_id else f"T_{t_id}"
            color = (0, 255, 0) if confirmed_id else (0, 0, 255)
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(output_frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Draw broadcast boxes
        for b_id, b_info in b_players.items():
            x1, y1, x2, y2 = map(int, b_info['bbox'])
            # Shift coordinates for combined frame
            cv2.rectangle(output_frame, (x1 + w_tacticam, y1), (x2 + w_tacticam, y2), (255, 0, 0), 2)
            cv2.putText(output_frame, f"B_{b_id}", (x1 + w_tacticam, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        output_video.write(output_frame)
        if frame_idx % 100 == 0:
            logging.info(f"Processed {frame_idx} frames. Confirmed matches: {len(id_matcher.confirmed_matches)}")

    logging.info("Releasing resources...")
    cap_broadcast.release()
    cap_tacticam.release()
    output_video.release()
    cv2.destroyAllWindows()
    logging.info("Processing complete. Output saved to: %s", OUTPUT_VIDEO_PATH)