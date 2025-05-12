import numpy as np
import time
import gym
from gym import spaces
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch


class ROIDetectionEnv(gym.Env):
    def __init__(self, dataset, crop_size=(640,640), time_limit=120):
        """
        Initialize the ROI Detection Environment.
        
        Args:
            dataset: ROIDataset instance for loading images and annotations
            crop_size: Size of the ROIs to place (width, height)
            time_limit: Time limit for an episode in seconds
        """
        super(ROIDetectionEnv, self).__init__()
        
        self.dataset = dataset
        self.image_size = dataset.image_size
        self.current_sample = None
        self.crop_size = crop_size
        self.bbox_size = None
        self.bboxes = []
        self.time_limit = time_limit
        self.start_time = None
        
        # Define action space:
        # 0-3: Move bbox (up, down, left, right)
        # 4: Place bbox
        # 5: Remove bbox
        # 6: End episode
        self.action_space = spaces.Discrete(7)

        self.shaping_coeff = 0.01  # Coefficient for shaping reward (from script 2)
        self.gamma_potential_shaping = 0.995 # Discount factor for potential shaping (from script 2)
        
        # Observation space: image and current bbox state
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(self.image_size[1], self.image_size[0], 3), dtype=np.uint8),
            'bbox_state': spaces.Box(low=0, high=1, shape=(100, 4), dtype=np.float32)  # max 100 bboxes
        })

        # Store optimal ROIs
        self.optimal_rois = None

    def reset(self):
        """Reset the environment with a new image"""
        # import time # Moved to top-level import
        self.start_time = time.time()
        
        try:
            self.current_sample = next(self.dataset)
        except StopIteration:
            # Dataset exhausted — restart it
            self.dataset.__iter__()  # Resets index and optionally reshuffles
            self.current_sample = next(self.dataset)

        self.current_image = self.current_sample['resized_image']

        # Calculate visual bbox size for the resized image
        self.bbox_size = self._get_bbox_size()
        
        # Process annotations that are taller than bbox size
        self._process_tall_annotations()
        
        # Calculate optimal ROIs using K-means
        self.optimal_rois = self._calculate_optimal_rois()

        # Reset bboxes
        self.bboxes = []

        # Initial bbox center
        if self.bbox_size is None or not (len(self.bbox_size) == 2): # Should not happen if _get_bbox_size is correct
             # Fallback if bbox_size is somehow not set, though _get_bbox_size should handle it
            print("Warning: bbox_size is None or invalid in reset. Defaulting current_bbox.")
            self.current_bbox = [0,0, self.crop_size[0] // 10, self.crop_size[1] // 10] # Arbitrary small default
        else:
            self.current_bbox = [
                (self.image_size[0] - self.bbox_size[0]) // 2,
                (self.image_size[1] - self.bbox_size[1]) // 2,
                self.bbox_size[0], self.bbox_size[1]
            ]
        return self._get_observation()

    def _process_tall_annotations(self):
        """Process annotations that are taller than the bbox size by cropping them"""
        if 'annotations' not in self.current_sample:
            return
            
        processed_annotations = []
        
        # Ensure bbox_size is valid before using its height component
        if self.bbox_size is None or len(self.bbox_size) < 2:
            print("Warning: bbox_size not properly initialized in _process_tall_annotations.")
            # Keep original annotations if bbox_size is invalid to avoid errors
            self.current_sample['annotations'] = self.current_sample.get('annotations', [])
            return

        for ann in self.current_sample['annotations']:
            bbox = ann['bbox']  # [x, y, w, h]
            
            if bbox[3] <= self.bbox_size[1]:
                processed_annotations.append(ann)
                continue
                
            top_part = ann.copy()
            top_part['bbox'] = [bbox[0], bbox[1], bbox[2], self.bbox_size[1]]
            processed_annotations.append(top_part)
            
        self.current_sample['annotations'] = processed_annotations

    def _get_action_mask(self) -> np.ndarray:
        """ Computes the action mask for the current state. """
        mask = np.ones(self.action_space.n, dtype=np.int8) 

        if self.bbox_size is None or self.current_bbox is None:
            print("Warning: bbox_size or current_bbox is None in _get_action_mask. Cannot mask actions reliably.")
            return mask
        if not (len(self.bbox_size) == 2 and len(self.current_bbox) == 4):
            print("Warning: bbox_size or current_bbox has unexpected format. Cannot mask actions reliably.")
            return mask

        epsilon = 1e-6 
        if self.current_bbox[1] <= epsilon: 
            mask[0] = 0 
        if self.current_bbox[1] >= self.image_size[1] - self.bbox_size[1] - epsilon: 
            mask[1] = 0 
        if self.current_bbox[0] <= epsilon: 
            mask[2] = 0 
        if self.current_bbox[0] >= self.image_size[0] - self.bbox_size[0] - epsilon: 
            mask[3] = 0 

        MAX_BBOXES = 100
        IDENTICAL_PLACEMENT_IOU_THRESHOLD = 0.99
        can_place = True
        if len(self.bboxes) >= MAX_BBOXES:
            can_place = False
        else:
            for existing_bbox in self.bboxes:
                try: 
                    iou = self._calculate_iou(self.current_bbox, existing_bbox)
                    if iou >= IDENTICAL_PLACEMENT_IOU_THRESHOLD:
                        can_place = False
                        break 
                except Exception as e:
                    print(f"Warning: Error calculating IoU in _get_action_mask: {e}")
                    pass 
        if not can_place:
            mask[4] = 0 

        if not self.bboxes: 
            mask[5] = 0 
        return mask

    def step(self, action):
        """Take a step in the environment based on the action (with reward shaping)"""
        # import time # Moved to top-level import
        
        # Calculate potential before taking action (for movement actions)
        old_potential = 0.0
        if action < 4: # Only calculate for move actions
             old_potential = self._potential_function(self.current_bbox)
        
        reward = 0.0 # Initialize step reward
        done = False
        info = {} 
        metrics = {} 

        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.time_limit:
            done = True
            # Final reward calculation will happen below
        else:
            if action < 4:  # Move bbox
                self._move_bbox(action)
                new_potential = self._potential_function(self.current_bbox)
                shaping_reward = self.gamma_potential_shaping * new_potential - old_potential
                reward += shaping_reward * self.shaping_coeff # Add shaping reward to step reward
            elif action == 4:  # Place bbox
                reward = self._place_bbox() # Get immediate reward from placing
            elif action == 5:  # Remove bbox
                reward = self._remove_bbox() # Get immediate reward from removing
            elif action == 6:  # End episode
                done = True
                # Final reward calculation will happen below
            else:
                print(f"Warning: Invalid action received: {action}")
                pass

        if done:
            try: 
                final_reward_value, metrics = self._calculate_final_reward()
                reward += final_reward_value # Add final reward to any step reward (e.g. if ended by action)
                metrics['time'] = {'elapsed': elapsed_time, 'limit': self.time_limit}
                info['metrics'] = metrics
            except Exception as e:
                print(f"Error during final reward calculation: {e}")
                reward += -10.0 # Fallback penalty

        try: 
            next_action_mask = self._get_action_mask()
            info['action_mask'] = next_action_mask
        except Exception as e:
            print(f"Error calculating action mask: {e}")
            info['action_mask'] = np.ones(self.action_space.n, dtype=np.int8)

        try: 
            observation = self._get_observation()
        except Exception as e:
            print(f"Error getting observation: {e}")
            observation = self.observation_space.sample() 

        return observation, reward, done, info
    
    def _get_observation(self):
        """Get the current observation"""
        image = self.current_image.copy()
        
        for bbox in self.bboxes:
            cv2.rectangle(image, 
                          (int(bbox[0]), int(bbox[1])), 
                          (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                          (0, 255, 0), 2)
        
        # Ensure current_bbox and its components are valid before drawing
        if self.current_bbox and len(self.current_bbox) == 4:
             cv2.rectangle(image,
                         (int(self.current_bbox[0]), int(self.current_bbox[1])),
                         (int(self.current_bbox[0] + self.current_bbox[2]), 
                          int(self.current_bbox[1] + self.current_bbox[3])),
                         (255, 0, 0), 2)
        
        bbox_state = np.zeros((100, 4), dtype=np.float32) # Ensure dtype for observation space
        for i, bbox in enumerate(self.bboxes):
            if i < 100:
                bbox_state[i] = [
                    bbox[0]/self.image_size[0], 
                    bbox[1]/self.image_size[1],
                    bbox[2]/self.image_size[0],
                    bbox[3]/self.image_size[1]
                ]
        
        return {
            'image': image.astype(np.uint8), # Ensure dtype for observation space
            'bbox_state': bbox_state
        }

    def _move_bbox(self, direction):
        """Move the current bounding box"""
        step_size = 8
        # Ensure bbox_size and current_bbox are valid before moving
        if self.bbox_size is None or len(self.bbox_size) != 2 or \
           self.current_bbox is None or len(self.current_bbox) != 4:
            print("Warning: bbox_size or current_bbox is invalid in _move_bbox. Skipping move.")
            return

        if direction == 0:  # Up
            self.current_bbox[1] = max(0, self.current_bbox[1] - step_size)
        elif direction == 1:  # Down
            self.current_bbox[1] = min(self.image_size[1] - self.bbox_size[1], 
                                     self.current_bbox[1] + step_size)
        elif direction == 2:  # Left
            self.current_bbox[0] = max(0, self.current_bbox[0] - step_size)
        elif direction == 3:  # Right
            self.current_bbox[0] = min(self.image_size[0] - self.bbox_size[0], 
                                     self.current_bbox[0] + step_size)

    def _place_bbox(self):
        """
        Place the current bounding box.
        Logic based on script 1 (safeguards, Stage 2 overlap adds box then penalizes).
        Reward values/scaling for IoU part from script 2.
        """
        MAX_BBOXES = 100
        IDENTICAL_PLACEMENT_IOU_THRESHOLD = 0.99
        current_bbox_to_place = self.current_bbox.copy()

        # Safeguard 1: Max boxes reached? (Logic from script 1)
        # This should ideally be caught by action mask, but as a safeguard:
        if len(self.bboxes) >= MAX_BBOXES:
            return 0.0 # Neutral reward as it's an invalid state not a strategic error

        # Safeguard 2: Placing exactly on top of existing? (Logic from script 1)
        for existing_bbox in self.bboxes:
            iou = self._calculate_iou(current_bbox_to_place, existing_bbox)
            if iou >= IDENTICAL_PLACEMENT_IOU_THRESHOLD:
                return -0.2 # Penalty from script 1 (consistent with script 2 overlap)

        # --- Overlap Check (Logic from Script 1's "Stage 2") ---
        # Script 1's "Stage 2" logic: if significant overlap, add box, then return penalty.
        overlap_threshold_script1 = 0.8 
        overlap_penalty_script1 = -0.2 

        for existing_bbox in self.bboxes: 
            iou_with_existing = self._calculate_iou(current_bbox_to_place, existing_bbox)
            if iou_with_existing > overlap_threshold_script1:
                self.bboxes.append(current_bbox_to_place) # Script 1 logic: Add box
                return overlap_penalty_script1          # Return penalty

        # If no significant overlap found by the loop above (it would have returned),
        # then add the box and calculate IoU-based reward.
        self.bboxes.append(current_bbox_to_place)

        # --- IoU-based reward (Reward values/scaling from Script 2) ---
        max_iou_with_optimal = 0.0
        if self.optimal_rois:
            for opt_roi in self.optimal_rois:
                iou = self._calculate_iou(current_bbox_to_place, opt_roi)
                if iou > max_iou_with_optimal:
                    max_iou_with_optimal = iou
        
        # Reward calculation using Script 2's values:
        if max_iou_with_optimal > 0: # Script 2 condition for positive reward
            reward = (max_iou_with_optimal * max_iou_with_optimal) * 10.0 # Script 2 reward scaling
            return reward
        else:
            return 0.0 # Script 2 neutral reward if no good match / IoU is 0
        
    def _remove_bbox(self):
        """
        Remove the last placed bounding box.
        Logic: Pop if exists.
        Rewards based on script 2's smart rewards.
        """
        if not self.bboxes:
            return -0.1  # Reward from script 2: Penalty for trying to remove when no bbox exists
        
        last_bbox = self.bboxes[-1]
        
        max_iou = 0.0
        if self.optimal_rois:
            for opt_roi in self.optimal_rois:
                iou = self._calculate_iou(last_bbox, opt_roi)
                max_iou = max(max_iou, iou)
        
        self.bboxes.pop() # Logic: Remove the bbox
        
        # Rewards from script 2 (adjusted condition for clarity):
        # Script 2 had `if max_iou < 0.0:`, which is unlikely for IoU.
        # Assuming it meant a very low IoU indicates a "low-quality" box.
        low_quality_threshold = 0.05 # Example threshold for a "bad" box
        if max_iou < low_quality_threshold: 
            return 0.2  # Reward from script 2 for removing a low-quality box
        else:
            # Penalty proportional to how good the box was (from script 2)
            return -max_iou * 2.0

    def _dist_to_nearest_unmatched_opt_roi(self, bbox):
        """
        Calculate the minimum Manhattan distance from the center of the current bbox
        to the center of any optimal ROI that hasn't been matched yet.
        (Using version from script 1 - identical logic to script 2)
        """
        if not self.optimal_rois:
            return 0.0
        
        bbox_cx = bbox[0] + bbox[2] / 2.0
        bbox_cy = bbox[1] + bbox[3] / 2.0
        
        matched_opt_rois_indices = [] # Store indices of matched optimal ROIs
        
        for placed_bbox in self.bboxes:
            best_iou = 0.0
            best_match_idx = -1 # Use -1 to indicate no match found yet for this placed_bbox
            
            for i, roi in enumerate(self.optimal_rois):
                if i in matched_opt_rois_indices: # Skip if this optimal ROI is already matched by another placed_bbox
                    continue
                iou = self._calculate_iou(placed_bbox, roi)
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = i
            
            if best_match_idx != -1 and best_iou > 0.1: # If a good match is found for placed_bbox
                if best_match_idx not in matched_opt_rois_indices: # ensure we don't add duplicates if multiple placed_bboxes match the same optimal_roi
                    matched_opt_rois_indices.append(best_match_idx)
        
        min_dist = float('inf')
        unmatched_found = False
        for i, roi in enumerate(self.optimal_rois):
            if i in matched_opt_rois_indices:
                continue
            unmatched_found = True
            roi_cx = roi[0] + roi[2] / 2.0
            roi_cy = roi[1] + roi[3] / 2.0
            dist = abs(bbox_cx - roi_cx) + abs(bbox_cy - roi_cy)
            min_dist = min(min_dist, dist)
        
        if not unmatched_found: # All optimal ROIs are matched
            return 0.0
        
        return min_dist if min_dist != float('inf') else 0.0


    def _potential_function(self, bbox):
        """
        Potential function for reward shaping.
        Returns a higher value when closer to an unmatched optimal ROI.
        (Using version from script 1 - slightly more robust jitter seed)
        """
        distance = self._dist_to_nearest_unmatched_opt_roi(bbox)
        
        jitter_seed = int(bbox[0] * 1000 + bbox[1])
        jitter_seed = abs(int(jitter_seed)) % (2**32)
        rng = np.random.RandomState(jitter_seed) # Corrected: was np.random.Randomstate
        jitter_scale = min(self.image_size) * 0.005  
        jitter = rng.normal(0, jitter_scale)
        
        return -distance + jitter

    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes"""
        box1_x1, box1_y1 = box1[0], box1[1]
        box1_x2, box1_y2 = box1[0] + box1[2], box1[1] + box1[3]
        
        box2_x1, box2_y1 = box2[0], box2[1]
        box2_x2, box2_y2 = box2[0] + box2[2], box2[1] + box2[3]
        
        x_left = max(box1_x1, box2_x1)
        y_top = max(box1_y1, box2_y1)
        x_right = min(box1_x2, box2_x2)
        y_bottom = min(box1_y2, box2_y2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        union_area = box1_area + box2_area - intersection_area
        
        if union_area == 0: # Avoid division by zero
            return 0.0
        return intersection_area / union_area

    def _scale_bbox_to_original(self, bbox):
        """Scale a bbox from the resized image back to the original image dimensions"""
        scale_factors = self.current_sample['scale_factors']
        original_bbox = [
            int(bbox[0] / scale_factors[0]),
            int(bbox[1] / scale_factors[1]),
            int(bbox[2] / scale_factors[0]),
            int(bbox[3] / scale_factors[1])
        ]
        return original_bbox

    def _calculate_optimal_rois(self):
        """
        KMeans-based ROI discovery under L∞ constraint.
        """
        annotations = self.current_sample.get('annotations', []) # Handle missing annotations
        if not annotations: # No annotations, place a single ROI in the center
            H, W = self.current_image.shape[:2]
            roi_w, roi_h = self.bbox_size if self.bbox_size else self.crop_size # Fallback
            half_w, half_h = roi_w / 2.0, roi_h / 2.0
            cx, cy = W / 2.0, H / 2.0
            return [[max(0, cx - half_w), max(0, cy - half_h), roi_w, roi_h]]


        H, W = self.current_image.shape[:2]
        # Ensure bbox_size is valid
        if self.bbox_size is None or len(self.bbox_size) != 2:
             print("Warning: bbox_size invalid in _calculate_optimal_rois. Using crop_size as fallback.")
             # Fallback to crop_size if bbox_size is not set, though it should be from reset
             # This might lead to ROIs not matching the agent's movable bbox size if crop_size is original scale
             # A better fallback might be to estimate from annotations if possible or use a default fraction of image size
             self.bbox_size = self.crop_size # This assumes crop_size is on the resized scale, which might be incorrect.
                                             # For safety, _get_bbox_size should always run first.
        roi_w, roi_h = self.bbox_size
        half_w, half_h = roi_w/2.0, roi_h/2.0
        
        def clamp(val, lo, hi):
            return max(lo, min(val, hi))
            
        pts = []
        for ann in annotations:
            x, y, w, h_ann = ann['bbox'] # Renamed h to h_ann to avoid conflict
            pts.append((x + w/2.0, y + h_ann/2.0))
        
        if not pts: # Should be caught by the first check, but as a safeguard
             cx, cy = W/2.0, H/2.0
             return [[max(0, cx - half_w), max(0, cy - half_h), roi_w, roi_h]]
        if len(pts) == 1:
            cx, cy = pts[0]
            cx = clamp(cx, half_w, W - half_w)
            cy = clamp(cy, half_h, H - half_h)
            return [[cx - half_w, cy - half_h, roi_w, roi_h]]
            
        pts = np.array(pts)
        
        best_k = len(pts)
        best_centers = pts # Default to each point being a center if no k works
        
        for k_clusters in range(1, len(pts)+1): # Renamed k to k_clusters
            if len(pts) < k_clusters:
                continue
            
            # Kmeans can fail with n_samples < n_clusters. Added explicit check.
            # n_init='auto' is recommended for scikit-learn >= 1.4, otherwise 10 is default.
            # For older versions, explicitly set n_init=10 or handle the warning.
            try:
                km = KMeans(n_clusters=k_clusters, random_state=0, n_init='auto').fit(pts)
            except ValueError as e:
                # Handle cases where KMeans might fail (e.g. not enough distinct samples for k_clusters)
                # print(f"KMeans failed for k={k_clusters} with error: {e}. Skipping this k.")
                if k_clusters == 1 and len(pts) >=1: # If k=1 fails, something is very wrong or pts is empty
                     # Fallback to average of all points for k=1 if it fails
                     center = np.mean(pts, axis=0, keepdims=True)
                     best_centers = center
                     best_k = 1
                     break # Found a fallback for k=1
                continue # Try next k or rely on default best_centers=pts

            centers = km.cluster_centers_
            labels = km.labels_
            
            good = True
            for ci in range(k_clusters):
                cx, cy = centers[ci]
                members = pts[labels == ci] # More efficient way to get members
                if not members.any(): 
                    continue
                
                max_dx = np.max(np.abs(members[:, 0] - cx)) if members.size > 0 else 0
                max_dy = np.max(np.abs(members[:, 1] - cy)) if members.size > 0 else 0

                if max_dx > half_w or max_dy > half_h:
                    good = False
                    break
            
            if good:
                best_k = k_clusters
                best_centers = centers
                break # Found the minimal k
        
        rois = []
        # Ensure best_centers is not None and is iterable
        if best_centers is not None and len(best_centers) > 0 :
            for (cx, cy) in best_centers:
                cx = clamp(cx, half_w, W - half_w)
                cy = clamp(cy, half_h, H - half_h)
                x0, y0 = cx - half_w, cy - half_h
                rois.append([x0, y0, roi_w, roi_h])
        else: # Fallback if best_centers is somehow empty or None
             cx_fallback, cy_fallback = W/2.0, H/2.0
             rois.append([max(0, cx_fallback - half_w), max(0, cy_fallback - half_h), roi_w, roi_h])

        return rois


    def _is_bbox_contained(self, bbox1, bbox2, threshold=0.8):
        """Check if bbox1 is mostly contained within bbox2"""
        bbox1_x1, bbox1_y1 = bbox1[0], bbox1[1]
        bbox1_x2, bbox1_y2 = bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]
        
        bbox2_x1, bbox2_y1 = bbox2[0], bbox2[1]
        bbox2_x2, bbox2_y2 = bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]
        
        x_left = max(bbox1_x1, bbox2_x1)
        y_top = max(bbox1_y1, bbox2_y1)
        x_right = min(bbox1_x2, bbox2_x2)
        y_bottom = min(bbox1_y2, bbox2_y2)
        
        if x_right < x_left or y_bottom < y_top:
            return False
                
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bbox1_area = (bbox1_x2 - bbox1_x1) * (bbox1_y2 - bbox1_y1)
        
        if bbox1_area == 0: # Avoid division by zero if annotation area is zero
             return False 
        return intersection_area / bbox1_area >= threshold

    def _calculate_annotation_coverage(self, rois, annotations):
        """Calculate how many annotations are properly covered by at least one ROI"""
        covered_count = 0
        if not annotations: # Handle case of no annotations
            return 1.0 
        total_annotations = len(annotations)
        
        for ann in annotations:
            ann_bbox = ann['bbox'] 
            for roi in rois:
                if self._is_bbox_contained(ann_bbox, roi):
                    covered_count += 1
                    break 
        
        return covered_count / total_annotations if total_annotations > 0 else 1.0


    def _calculate_roi_matching(self, placed_rois, optimal_rois):
        """Calculate how well placed ROIs match optimal ROIs using IoU"""
        if not optimal_rois or not placed_rois:
            return 0.0
        
        total_iou = 0.0
        
        # Create a list to track if an optimal ROI has been matched by a placed ROI
        # to avoid double-counting when multiple placed ROIs cover the same optimal ROI.
        # Instead, for each optimal ROI, find its best match among placed ROIs.
        
        for opt_roi in optimal_rois:
            best_iou_for_this_opt_roi = 0.0
            for placed_roi in placed_rois:
                iou = self._calculate_iou(opt_roi, placed_roi)
                best_iou_for_this_opt_roi = max(best_iou_for_this_opt_roi, iou)
            total_iou += best_iou_for_this_opt_roi # Sum of best IoUs for each optimal ROI
        
        return total_iou / len(optimal_rois) if optimal_rois else 0.0

    def _calculate_roi_overlap_penalty(self, rois):
        """Calculate penalty for excessive overlap between ROIs"""
        if len(rois) <= 1:
            return 0.0
        
        total_penalty = 0.0
        num_pairs = 0
        for i in range(len(rois)):
            for j in range(i+1, len(rois)):
                iou = self._calculate_iou(rois[i], rois[j])
                if iou > 0.3: # Only penalize IoU above 0.3 (from script 1)
                    # Script 1 had: total_penalty += (iou - 0.1)
                    # This can lead to large penalties.
                    # Let's use a simpler penalty: just sum up significant overlaps.
                    total_penalty += iou 
                num_pairs +=1
        
        # Normalize penalty by number of pairs if desired, or cap it.
        # Script 1 capped at 1.0: return min(1.0, total_penalty)
        # If using sum of IoUs, this cap is important.
        return min(1.0, total_penalty) 

    def _calculate_final_reward(self):
        """
        Calculate final reward.
        Logic structure from script 1. Reward components/weights from script 2.
        """
        if not self.bboxes:
            return -10.0, {"metrics": "No ROIs placed"} 

        if self.optimal_rois is None:
            self.optimal_rois = self._calculate_optimal_rois()
        
        annotations = self.current_sample.get('annotations', []) # Handle missing annotations
        coverage_score = self._calculate_annotation_coverage(self.bboxes, annotations)
        roi_matching_score = self._calculate_roi_matching(self.bboxes, self.optimal_rois)
        
        optimal_count = len(self.optimal_rois) if self.optimal_rois else 0
        placed_count = len(self.bboxes)

        # Reward calculation from Script 2
        coverage_weight = 70.0
        matching_weight = 30.0
        
        final_reward = (
            coverage_score * coverage_weight +
            roi_matching_score * matching_weight
        )
        
        metrics = {
            'optimal_rois': self.optimal_rois,
            'coverage_score': coverage_score,
            'roi_matching_score': roi_matching_score,
            'optimal_count': optimal_count,
            'placed_count': placed_count
        }
        
        return final_reward, metrics
        
    def _get_bbox_size(self):
        """
        Calculate the bbox size (width, height) in the resized image
        that corresponds to crop_size in the original image.
        """
        if self.current_sample is None or 'scale_factors' not in self.current_sample:
            # This can happen if reset() hasn't fully completed or dataset is malformed
            # Fallback to a fraction of image size or a default if image_size is also not ready
            print("Warning: current_sample or scale_factors not available in _get_bbox_size. Using default.")
            if self.image_size and self.image_size[0] > 0 and self.image_size[1] > 0:
                 # Default to a small fraction of the image if possible
                 return int(self.image_size[0] * 0.1), int(self.image_size[1] * 0.1)
            return self.crop_size # Fallback, though this might be original scale
        
        scale_w, scale_h = self.current_sample['scale_factors']
        crop_w, crop_h = self.crop_size
        return int(crop_w * scale_w), int(crop_h * scale_h)

    def render(self, mode='rgb_array'):
        """Render the current state of the environment"""
        if self.current_image is None:
            # Create a dummy black image if current_image is None
            dummy_h = self.image_size[1] if self.image_size else 256
            dummy_w = self.image_size[0] if self.image_size else 256
            image = np.zeros((dummy_h, dummy_w, 3), dtype=np.uint8)
            # Add text indicating no image
            cv2.putText(image, "No image loaded", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            image = self.current_image.copy()
        
        if 'annotations' in self.current_sample:
            for ann in self.current_sample['annotations']:
                bbox = ann['bbox']
                cv2.rectangle(image, 
                              (int(bbox[0]), int(bbox[1])), 
                              (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                              (0, 255, 255), 1) # Yellow for GT annotations
        
        if self.optimal_rois:
            for roi in self.optimal_rois:
                cv2.rectangle(image,
                              (int(roi[0]), int(roi[1])),
                              (int(roi[0] + roi[2]), int(roi[1] + roi[3])),
                              (0, 0, 255), 1)  # Red for optimal ROIs
                
        for bbox_placed in self.bboxes: # Renamed to avoid conflict
            cv2.rectangle(image, 
                          (int(bbox_placed[0]), int(bbox_placed[1])), 
                          (int(bbox_placed[0] + bbox_placed[2]), int(bbox_placed[1] + bbox_placed[3])),
                          (0, 255, 0), 2) # Green for placed bboxes
        
        if self.current_bbox and len(self.current_bbox) == 4: # Check if current_bbox is valid
            cv2.rectangle(image,
                        (int(self.current_bbox[0]), int(self.current_bbox[1])),
                        (int(self.current_bbox[0] + self.current_bbox[2]), 
                         int(self.current_bbox[1] + self.current_bbox[3])),
                        (255, 0, 0), 2) # Blue for current movable bbox
                     
        if mode == 'rgb_array':
            return image
        elif mode == 'human':
            cv2.imshow('ROI Detection Environment', image)
            cv2.waitKey(1)
            return image # Also return image for consistency if needed elsewhere
            
    def visualize_reward_landscape(self, output_path="reward_landscape.jpg"):
        """
        Create a visualization of the reward landscape.
        """
        # import matplotlib.pyplot as plt # Moved to top-level
        # from matplotlib import cm      # Moved to top-level
        # import numpy as np            # Moved to top-level
        
        if self.current_image is None or self.bbox_size is None:
            print("Cannot visualize reward landscape: current image or bbox_size is not set.")
            return None

        matched_opt_rois_indices = [] # Store indices
        if self.optimal_rois:
            # Determine which optimal ROIs are matched by placed_bboxes
            # This logic is similar to _dist_to_nearest_unmatched_opt_roi's matching part
            temp_matched_indices = [] # To avoid modifying list while iterating if complex logic were used
            for placed_bbox_vis in self.bboxes: # Renamed to avoid conflict
                best_iou_vis = 0.0
                best_match_idx_vis = -1
                for i, roi_vis in enumerate(self.optimal_rois):
                     if i in temp_matched_indices: continue # Already matched by another placed box
                     iou_vis = self._calculate_iou(placed_bbox_vis, roi_vis)
                     if iou_vis > best_iou_vis:
                         best_iou_vis = iou_vis
                         best_match_idx_vis = i
                if best_match_idx_vis != -1 and best_iou_vis > 0.1:
                     if best_match_idx_vis not in temp_matched_indices:
                          temp_matched_indices.append(best_match_idx_vis)
            matched_opt_rois_indices = temp_matched_indices

        step_vis = max(1, min(self.image_size) // 100) 
        width_img, height_img = self.image_size # Renamed to avoid conflict
        
        rows_map = height_img // step_vis
        cols_map = width_img // step_vis
        potential_map = np.zeros((rows_map, cols_map))
        
        # Temporarily clear self.bboxes for landscape calculation if it's based on current_bbox to *any* unmatched
        # No, _dist_to_nearest_unmatched_opt_roi already considers current self.bboxes
        # The landscape should reflect the potential for the *next* placement given current state.

        for r_idx in range(rows_map): # r_idx for row index
            for c_idx in range(cols_map): # c_idx for col index
                i_coord = r_idx * step_vis # y-coordinate
                j_coord = c_idx * step_vis # x-coordinate
                                 
                test_bbox = [
                    max(0, j_coord - self.bbox_size[0] // 2),
                    max(0, i_coord - self.bbox_size[1] // 2),
                    self.bbox_size[0],
                    self.bbox_size[1]
                ]
                
                # Calculate potential without jitter for visualization clarity
                # Create a temporary list of bboxes that would exist if test_bbox was the current_bbox
                # and we are evaluating its potential based on existing self.bboxes
                potential = -self._dist_to_nearest_unmatched_opt_roi(test_bbox) 
                potential_map[r_idx, c_idx] = potential # Correct indexing
        
        potential_min = np.min(potential_map)
        potential_max = np.max(potential_map)
        if potential_max > potential_min: # Avoid division by zero if flat landscape
            potential_map_normalized = (potential_map - potential_min) / (potential_max - potential_min)
        elif potential_max == potential_min and potential_max != 0 : # Flat but not zero
            potential_map_normalized = np.ones_like(potential_map) * 0.5 # Mid-gray
        else: # Flat and zero, or all same value
            potential_map_normalized = np.zeros_like(potential_map)

        
        plt.figure(figsize=(12, 10)) # Adjusted size
        
        plt.imshow(cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB))
        
        plt.imshow(potential_map_normalized, cmap=cm.jet, alpha=0.5, interpolation='bilinear', 
                   extent=[0, width_img, height_img, 0]) 
        
        if self.optimal_rois:
            for i, roi_p in enumerate(self.optimal_rois): # roi_p for roi_plot
                edge_color = 'g' if i in matched_opt_rois_indices else 'r'
                plt.gca().add_patch(plt.Rectangle(
                    (roi_p[0], roi_p[1]), roi_p[2], roi_p[3], 
                    linewidth=2, edgecolor=edge_color, facecolor='none'
                ))
        
        for bbox_p in self.bboxes: # bbox_p for bbox_plot
            plt.gca().add_patch(plt.Rectangle(
                (bbox_p[0], bbox_p[1]), bbox_p[2], bbox_p[3], 
                linewidth=2, edgecolor='lime', facecolor='none', linestyle='--' # Changed color for placed
            ))
        
        cbar = plt.colorbar(mappable=cm.ScalarMappable(cmap=cm.jet), ax=plt.gca()) # Ensure colorbar matches imshow
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels([f'{potential_min:.2f} (Low)', 'Medium', f'{potential_max:.2f} (High)'])
        cbar.set_label('Potential Value (Higher is Better, towards Unmatched Optimal ROI)')
        
        plt.title('Reward Shaping Potential Landscape')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        
        # from matplotlib.patches import Patch # Moved to top-level
        legend_elements = [
            Patch(facecolor='none', edgecolor='r', linewidth=2, label='Unmatched Optimal ROI'),
            Patch(facecolor='none', edgecolor='g', linewidth=2, label='Matched Optimal ROI (by a placed bbox)'),
            Patch(facecolor='none', edgecolor='lime', linewidth=2, linestyle='--', label='Placed Bbox')
        ]
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.05, 1)) # Adjust legend position
        
        plt.tight_layout() # Adjust layout
        plt.savefig(output_path, dpi=300) # Removed bbox_inches='tight' as tight_layout is used
        plt.close()
        
        print(f"Reward landscape visualization saved to {output_path}")
        return output_path