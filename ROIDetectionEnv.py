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

        self.shaping_coeff = 0.01  # Coefficient for shaping reward
        self.gamma_potential_shaping = 0.995 # Discount factor for potential shaping
        
        # Observation space: image and current bbox state
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(self.image_size[1], self.image_size[0], 3), dtype=np.uint8),
            'bbox_state': spaces.Box(low=0, high=1, shape=(100, 4), dtype=np.float32)  # max 100 bboxes
        })

        # Store optimal ROIs
        self.optimal_rois = None

    def reset(self):
        """Reset the environment with a new image"""
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
        if self.bbox_size is None or not (len(self.bbox_size) == 2): 
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
        mask = np.ones(self.action_space.n, dtype=np.int8) # Start with all actions allowed (1 = True)

        if self.bbox_size is None or self.current_bbox is None:
            print("Warning: bbox_size or current_bbox is None in _get_action_mask. Cannot mask actions reliably.")
            return mask # Return all actions enabled as a fallback
        if not (len(self.bbox_size) == 2 and len(self.current_bbox) == 4):
            print("Warning: bbox_size or current_bbox has unexpected format. Cannot mask actions reliably.")
            return mask # Return all actions enabled

        # --- Check Movement Boundaries ---
        epsilon = 1e-6 # Optional tolerance for floating point comparisons
        if self.current_bbox[1] <= epsilon: # At top boundary
            mask[0] = 0 # Disable Move Up
        if self.current_bbox[1] >= self.image_size[1] - self.bbox_size[1] - epsilon: # At bottom boundary
            mask[1] = 0 # Disable Move Down
        if self.current_bbox[0] <= epsilon: # At left boundary
            mask[2] = 0 # Disable Move Left
        if self.current_bbox[0] >= self.image_size[0] - self.bbox_size[0] - epsilon: # At right boundary
            mask[3] = 0 # Disable Move Right

        # --- Check Place Bbox Conditions (Action 4) ---
        MAX_BBOXES = 100
        IDENTICAL_PLACEMENT_IOU_THRESHOLD = 0.99
        can_place = True
        if len(self.bboxes) >= MAX_BBOXES: # Condition 1: Max boxes limit reached?
            can_place = False
        else:
            # Condition 2: Trying to place exactly on top of an existing one?
            for existing_bbox in self.bboxes:
                try: 
                    iou = self._calculate_iou(self.current_bbox, existing_bbox)
                    if iou >= IDENTICAL_PLACEMENT_IOU_THRESHOLD:
                        can_place = False
                        break # Found an identical placement, no need to check further
                except Exception as e:
                    print(f"Warning: Error calculating IoU in _get_action_mask: {e}")
                    pass # Or ignore this specific check, or decide to disable placement for safety
        if not can_place:
            mask[4] = 0 # Disable Place Bbox

        # --- Check Remove Bbox Condition (Action 5) ---
        if not self.bboxes: # If the list of placed bboxes is empty
            mask[5] = 0 # Disable Remove Bbox

        # --- End Episode (Action 6) ---
        # This action is typically always allowed (mask[6] remains 1)
        return mask

    def action_masks(self) -> np.ndarray:
        """
        Returns the valid action mask for the current state.
        This is required by MaskablePPO.
        """
        return self._get_action_mask()

    def step(self, action):
        """Take a step in the environment based on the action (with reward shaping)"""
        old_potential = 0.0
        if action < 4: # Only calculate potential for move actions
             old_potential = self._potential_function(self.current_bbox)
        
        reward = 0.0 # Initialize step reward
        done = False
        info = {} 
        metrics = {} 

        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.time_limit:
            done = True
            # Final reward calculation will happen below if episode ends due to time limit
        else:
            # Execute the chosen action
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
                # Final reward calculation will happen below if episode ends by agent's choice
            else:
                print(f"Warning: Invalid action received: {action}")
                # No explicit penalty for invalid action here, as it should be masked
                pass

        # Calculate final reward if episode is done (either by time or action 6)
        if done:
            try: 
                final_reward_value, metrics_from_final_reward = self._calculate_final_reward()
                reward += final_reward_value # Add final reward to any existing step reward
                metrics.update(metrics_from_final_reward) # Merge metrics
                metrics['time'] = {'elapsed': elapsed_time, 'limit': self.time_limit}
                info['metrics'] = metrics
            except Exception as e:
                print(f"Error during final reward calculation: {e}")
                reward += -10.0 # Fallback penalty if final reward calculation fails
                info['metrics'] = metrics # metrics might be partially filled or empty

        # Get action mask for the *next* state (required by some SB3 algorithms, good practice)
        # MaskablePPO will use the action_masks() method directly during its own step.
        # However, returning it in info is still a common pattern.
        try: 
            next_action_mask = self._get_action_mask() # or self.action_masks()
            info['action_mask'] = next_action_mask
        except Exception as e:
            print(f"Error calculating action mask for info: {e}")
            info['action_mask'] = np.ones(self.action_space.n, dtype=np.int8) # Default: all actions enabled

        # Get observation for the next state
        try: 
            observation = self._get_observation()
        except Exception as e:
            print(f"Error getting observation: {e}")
            observation = self.observation_space.sample() # Return a sample observation as fallback

        return observation, reward, done, info
    
    def _get_observation(self):
        """Get the current observation"""
        image = self.current_image.copy()
        
        # Draw all placed bboxes
        for bbox in self.bboxes:
            cv2.rectangle(image, 
                          (int(bbox[0]), int(bbox[1])), 
                          (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                          (0, 255, 0), 2) # Green
        
        # Ensure current_bbox and its components are valid before drawing
        if self.current_bbox and len(self.current_bbox) == 4:
             cv2.rectangle(image,
                         (int(self.current_bbox[0]), int(self.current_bbox[1])),
                         (int(self.current_bbox[0] + self.current_bbox[2]), 
                          int(self.current_bbox[1] + self.current_bbox[3])),
                         (255, 0, 0), 2) # Blue (movable)
        
        bbox_state = np.zeros((100, 4), dtype=np.float32) # Ensure dtype for observation space
        for i, bbox in enumerate(self.bboxes):
            if i < 100: # Max 100 bboxes
                bbox_state[i] = [
                    bbox[0]/self.image_size[0], 
                    bbox[1]/self.image_size[1],
                    bbox[2]/self.image_size[0],
                    bbox[3]/self.image_size[1]
                ] # Normalize bbox coordinates
        
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
        Logic includes safeguards and overlap checks.
        Reward is based on IoU with optimal ROIs.
        """
        MAX_BBOXES = 100
        IDENTICAL_PLACEMENT_IOU_THRESHOLD = 0.99
        current_bbox_to_place = self.current_bbox.copy()

        # Safeguard 1: Max boxes reached?
        if len(self.bboxes) >= MAX_BBOXES:
            return 0.0 # Neutral reward as this should ideally be caught by action mask

        # Safeguard 2: Placing almost exactly on top of an existing bbox?
        for existing_bbox in self.bboxes:
            iou = self._calculate_iou(current_bbox_to_place, existing_bbox)
            if iou >= IDENTICAL_PLACEMENT_IOU_THRESHOLD:
                return -0.2 # Penalty for identical placement

        # Overlap Check (as per original logic from script 1):
        # If significant overlap, add box, then return penalty.
        overlap_threshold_significant = 0.8 
        overlap_penalty_significant = -0.2 

        for existing_bbox in self.bboxes: 
            iou_with_existing = self._calculate_iou(current_bbox_to_place, existing_bbox)
            if iou_with_existing > overlap_threshold_significant:
                self.bboxes.append(current_bbox_to_place) # Add box despite overlap
                return overlap_penalty_significant          # Return penalty

        # If no significant overlap (that triggers immediate return) was found, add the box.
        self.bboxes.append(current_bbox_to_place)

        # IoU-based reward calculation (rewards from script 2's logic)
        max_iou_with_optimal = 0.0
        if self.optimal_rois:
            for opt_roi in self.optimal_rois:
                iou = self._calculate_iou(current_bbox_to_place, opt_roi)
                if iou > max_iou_with_optimal:
                    max_iou_with_optimal = iou
        
        if max_iou_with_optimal > 0: 
            reward = (max_iou_with_optimal * max_iou_with_optimal) * 10.0 
            return reward
        else:
            return 0.0 # Neutral reward if no good match / IoU is 0
        
    def _remove_bbox(self):
        """
        Remove the last placed bounding box.
        Rewards encourage removing "bad" boxes and penalize removing "good" boxes.
        """
        if not self.bboxes:
            return -0.1  # Penalty for trying to remove when no bbox exists
        
        last_bbox = self.bboxes[-1]
        
        max_iou = 0.0 # Max IoU of the removed box with any optimal ROI
        if self.optimal_rois:
            for opt_roi in self.optimal_rois:
                iou = self._calculate_iou(last_bbox, opt_roi)
                max_iou = max(max_iou, iou)
        
        self.bboxes.pop() # Remove the bbox
        
        low_quality_threshold = 0.05 # Threshold to consider a box "low quality"
        if max_iou < low_quality_threshold: 
            return 0.2  # Reward for removing a low-quality box
        else:
            return -max_iou * 2.0 # Penalty for removing a good box, proportional to its quality
            
    def _dist_to_nearest_unmatched_opt_roi(self, bbox):
        """
        Calculate the minimum Manhattan distance from the center of the current bbox
        to the center of any optimal ROI that hasn't been matched yet by a *placed* bbox.
        """
        if not self.optimal_rois:
            return 0.0
        
        bbox_cx = bbox[0] + bbox[2] / 2.0
        bbox_cy = bbox[1] + bbox[3] / 2.0
        
        # Identify indices of optimal ROIs already "matched" by one of the self.bboxes
        matched_opt_rois_indices = [] 
        temp_optimal_rois_for_matching = list(enumerate(self.optimal_rois))

        for placed_bbox_in_list in self.bboxes: # Check against already placed bboxes
            best_iou_for_placed = 0.0
            best_match_opt_idx = -1
            
            for i, opt_roi_data in temp_optimal_rois_for_matching:
                if i in matched_opt_rois_indices: # If this optimal ROI is already claimed
                    continue
                iou = self._calculate_iou(placed_bbox_in_list, opt_roi_data)
                if iou > best_iou_for_placed:
                    best_iou_for_placed = iou
                    best_match_opt_idx = i # Store index of the optimal ROI
            
            if best_match_opt_idx != -1 and best_iou_for_placed > 0.1: # If a good match is found
                if best_match_opt_idx not in matched_opt_rois_indices:
                    matched_opt_rois_indices.append(best_match_opt_idx)
        
        min_dist = float('inf')
        unmatched_roi_exists = False
        for i, roi in enumerate(self.optimal_rois):
            if i in matched_opt_rois_indices: # Skip optimal ROIs that are already matched
                continue
            unmatched_roi_exists = True # Found at least one unmatched optimal ROI
            roi_cx = roi[0] + roi[2] / 2.0
            roi_cy = roi[1] + roi[3] / 2.0
            dist = abs(bbox_cx - roi_cx) + abs(bbox_cy - roi_cy) # Manhattan distance
            min_dist = min(min_dist, dist)
        
        if not unmatched_roi_exists: # All optimal ROIs are currently matched by placed bboxes
            return 0.0 # No distance penalty needed, or agent should place to cover new areas if logic allows
        
        return min_dist if min_dist != float('inf') else 0.0 # Return 0 if somehow no unmatched ROIs found but min_dist is inf

    def _potential_function(self, bbox):
        """
        Potential function for reward shaping.
        Returns a higher value when closer to an unmatched optimal ROI. Includes jitter.
        """
        distance = self._dist_to_nearest_unmatched_opt_roi(bbox)
        
        # Seeded random for consistent jitter based on bbox position
        jitter_seed = int(bbox[0] * 1000 + bbox[1]) # Simple hash of position
        jitter_seed = abs(int(jitter_seed)) % (2**32) # Ensure positive seed within RNG limits
        rng = np.random.RandomState(jitter_seed) 
        jitter_scale = min(self.image_size[0], self.image_size[1]) * 0.005  # Reduced jitter scale
        jitter = rng.normal(0, jitter_scale)
        
        # Potential is higher (less negative) when distance is smaller
        return -distance + jitter

    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes [x, y, w, h]"""
        box1_x1, box1_y1 = box1[0], box1[1]
        box1_x2, box1_y2 = box1[0] + box1[2], box1[1] + box1[3]
        
        box2_x1, box2_y1 = box2[0], box2[1]
        box2_x2, box2_y2 = box2[0] + box2[2], box2[1] + box2[3]
        
        # Calculate intersection coordinates
        x_left = max(box1_x1, box2_x1)
        y_top = max(box1_y1, box2_y1)
        x_right = min(box1_x2, box2_x2)
        y_bottom = min(box1_y2, box2_y2)
        
        if x_right < x_left or y_bottom < y_top: # No overlap
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        box1_area = box1[2] * box1[3] # w * h
        box2_area = box2[2] * box2[3] # w * h
        
        union_area = box1_area + box2_area - intersection_area
        
        if union_area == 0: # Avoid division by zero
            return 0.0
        
        iou = intersection_area / union_area
        return iou

    def _scale_bbox_to_original(self, bbox):
        """Scale a bbox from the resized image back to the original image dimensions"""
        if 'scale_factors' not in self.current_sample: return bbox # Should not happen
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
        Determines a "reasonable" set of target ROIs based on annotation clustering.
        """
        annotations = self.current_sample.get('annotations', [])
        
        if self.bbox_size is None or len(self.bbox_size) != 2:
             print("Critical Warning: bbox_size is not set in _calculate_optimal_rois. Optimal ROIs might be incorrect.")
             # Attempt a fallback, but this indicates an issue in the reset/init flow
             self.bbox_size = self._get_bbox_size() # Try to get it again
             if self.bbox_size is None: # If still None, use a very rough default
                 self.bbox_size = (self.crop_size[0]*0.1, self.crop_size[1]*0.1)


        roi_w, roi_h = self.bbox_size
        half_w, half_h = roi_w / 2.0, roi_h / 2.0
        H, W = self.current_image.shape[:2]

        if not annotations: # If no annotations, create a single ROI in the center of the image
            cx, cy = W / 2.0, H / 2.0
            # Clamp to ensure ROI is within image boundaries
            x0 = max(0, cx - half_w)
            y0 = max(0, cy - half_h)
            # Ensure width/height don't exceed image dimensions if ROI is placed at edge
            clamped_w = min(roi_w, W - x0)
            clamped_h = min(roi_h, H - y0)
            return [[x0, y0, clamped_w, clamped_h]]

        def clamp_center(val, half_dim, total_dim): # Clamps center coordinate
            return max(half_dim, min(val, total_dim - half_dim))
            
        pts = [] # Collect annotation centers
        for ann in annotations:
            x, y, w_ann, h_ann = ann['bbox']
            pts.append((x + w_ann / 2.0, y + h_ann / 2.0))
        
        if not pts: # Should have been caught by `if not annotations`
             cx, cy = W/2.0, H/2.0
             x0 = max(0, cx - half_w); y0 = max(0, cy - half_h)
             return [[x0, y0, min(roi_w, W-x0), min(roi_h, H-y0)]]
        if len(pts) == 1: # Single annotation, create one ROI centered on it
            cx, cy = pts[0]
            cx = clamp_center(cx, half_w, W)
            cy = clamp_center(cy, half_h, H)
            x0 = cx - half_w; y0 = cy - half_h
            return [[x0, y0, roi_w, roi_h]] # Assuming roi_w, roi_h fit
            
        pts_np = np.array(pts)
        
        best_k = len(pts_np) # Max possible clusters is number of points
        best_centers = pts_np # Default: each annotation is its own cluster center

        for k_val in range(1, len(pts_np) + 1): 
            if len(pts_np) < k_val: # Cannot have more clusters than points
                continue
            
            try: # n_init='auto' is default in newer sklearn, otherwise set to 10
                km = KMeans(n_clusters=k_val, random_state=0, n_init='auto').fit(pts_np)
            except ValueError: # Kmeans can fail if k_val is too large for distinct points
                if k_val == 1 and len(pts_np) >=1: # Fallback for k=1 if it fails
                     best_centers = np.mean(pts_np, axis=0, keepdims=True)
                     best_k = 1; break 
                continue 

            current_centers = km.cluster_centers_
            labels = km.labels_
            
            is_good_k = True # Assume this k is good until proven otherwise
            for cluster_idx in range(k_val):
                cluster_center_x, cluster_center_y = current_centers[cluster_idx]
                member_points = pts_np[labels == cluster_idx]
                
                if not member_points.any(): # Skip empty clusters
                    continue
                
                # Check L-infinity radius for this cluster
                max_delta_x = np.max(np.abs(member_points[:, 0] - cluster_center_x))
                max_delta_y = np.max(np.abs(member_points[:, 1] - cluster_center_y))

                if max_delta_x > half_w or max_delta_y > half_h: # If any point is outside ROI half-dims
                    is_good_k = False
                    break # This k is not good, try next k
            
            if is_good_k: # Found the minimal k that satisfies the L-infinity constraint
                best_k = k_val
                best_centers = current_centers
                break 
        
        final_rois = []
        if best_centers is not None and len(best_centers) > 0 :
            for (center_x, center_y) in best_centers:
                # Clamp cluster centers so the resulting ROI is within image bounds
                clamped_center_x = clamp_center(center_x, half_w, W)
                clamped_center_y = clamp_center(center_y, half_h, H)
                
                x0 = clamped_center_x - half_w
                y0 = clamped_center_y - half_h
                # Ensure x0,y0 are not negative due to clamping very close to edge
                x0 = max(0, x0)
                y0 = max(0, y0)

                final_rois.append([x0, y0, roi_w, roi_h])
        else: # Fallback if best_centers is empty (should not happen if annotations exist)
             cx_fb, cy_fb = W/2.0, H/2.0
             x0_fb = max(0, cx_fb - half_w); y0_fb = max(0, cy_fb - half_h)
             final_rois.append([x0_fb, y0_fb, min(roi_w, W-x0_fb), min(roi_h, H-y0_fb)])
        return final_rois

    def _is_bbox_contained(self, bbox1_ann, bbox2_roi, threshold=0.8):
        """Check if bbox1_ann (annotation) is mostly contained within bbox2_roi (ROI)"""
        b1_x1, b1_y1, b1_w, b1_h = bbox1_ann
        b1_x2, b1_y2 = b1_x1 + b1_w, b1_y1 + b1_h
        
        b2_x1, b2_y1, b2_w, b2_h = bbox2_roi
        b2_x2, b2_y2 = b2_x1 + b2_w, b2_y1 + b2_h
        
        # Calculate intersection area
        inter_x1 = max(b1_x1, b2_x1)
        inter_y1 = max(b1_y1, b2_y1)
        inter_x2 = min(b1_x2, b2_x2)
        inter_y2 = min(b1_y2, b2_y2)
        
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        intersection_area = inter_w * inter_h
        
        if intersection_area == 0: # No overlap
            return False
                
        bbox1_area = b1_w * b1_h
        
        if bbox1_area == 0: # Avoid division by zero if annotation area is zero
             return False 
        # Return true if intersection covers at least `threshold` of bbox1_ann's area
        return (intersection_area / bbox1_area) >= threshold

    def _calculate_annotation_coverage(self, placed_rois, annotations):
        """Calculate fraction of annotations properly covered by at least one placed ROI"""
        if not annotations: # If no annotations to cover, coverage is perfect
            return 1.0 
        
        covered_count = 0
        total_annotations = len(annotations)
        
        for ann in annotations:
            ann_bbox = ann['bbox'] 
            for roi in placed_rois:
                if self._is_bbox_contained(ann_bbox, roi): # ann_bbox is contained in roi
                    covered_count += 1
                    break # This annotation is covered, move to the next annotation
        
        return covered_count / total_annotations if total_annotations > 0 else 1.0

    def _calculate_roi_matching(self, placed_rois, optimal_rois_list):
        """Calculate how well placed ROIs match optimal_rois_list using IoU"""
        if not optimal_rois_list or not placed_rois:
            return 0.0
        
        total_max_iou_sum = 0.0
        
        # For each optimal ROI, find the placed ROI that best matches it
        for opt_roi in optimal_rois_list:
            best_iou_for_this_optimal_roi = 0.0
            for p_roi in placed_rois: # p_roi for placed_roi
                iou = self._calculate_iou(opt_roi, p_roi) # Order matters if one IoU def is asymmetric
                best_iou_for_this_optimal_roi = max(best_iou_for_this_optimal_roi, iou)
            total_max_iou_sum += best_iou_for_this_optimal_roi
        
        # Average of the best IoUs found for each optimal ROI
        return total_max_iou_sum / len(optimal_rois_list)

    def _calculate_roi_overlap_penalty(self, rois_list): # Renamed to avoid conflict
        """Calculate penalty for excessive overlap between placed ROIs"""
        if len(rois_list) <= 1: # No overlap if 0 or 1 ROI
            return 0.0
        
        total_overlap_iou = 0.0
        num_pairs = 0
        for i in range(len(rois_list)):
            for j in range(i + 1, len(rois_list)):
                iou = self._calculate_iou(rois_list[i], rois_list[j])
                if iou > 0.3: # Only penalize significant IoU (threshold from original script 1)
                    total_overlap_iou += iou # Summing up significant overlaps
                num_pairs += 1
        
        # Capping the penalty at 1.0 (from original script 1)
        return min(1.0, total_overlap_iou) 

    def _calculate_final_reward(self):
        """
        Calculate final reward based on annotation coverage and ROI matching.
        Weights and components from script 2's reward logic.
        """
        if not self.bboxes: # No ROIs placed
            return -10.0, {"metrics": "No ROIs placed"} 

        if self.optimal_rois is None: # Should be calculated in reset
            self.optimal_rois = self._calculate_optimal_rois()
        
        annotations = self.current_sample.get('annotations', []) 
        coverage_score = self._calculate_annotation_coverage(self.bboxes, annotations)
        roi_matching_score = self._calculate_roi_matching(self.bboxes, self.optimal_rois)
        
        optimal_count = len(self.optimal_rois) if self.optimal_rois else 0
        placed_count = len(self.bboxes)

        # Final reward weights (from script 2's logic)
        coverage_weight = 70.0
        matching_weight = 30.0
        
        final_reward_val = ( # Renamed to avoid conflict
            coverage_score * coverage_weight +
            roi_matching_score * matching_weight
        )
        
        calculated_metrics = { # Renamed to avoid conflict
            'optimal_rois_list': self.optimal_rois, # Renamed key
            'coverage_score': coverage_score,
            'roi_matching_score': roi_matching_score,
            'optimal_count': optimal_count,
            'placed_count': placed_count
        }
        
        return final_reward_val, calculated_metrics
        
    def _get_bbox_size(self):
        """
        Calculate the bbox size (width, height) in the resized image
        that corresponds to crop_size in the original image.
        """
        if self.current_sample is None or 'scale_factors' not in self.current_sample:
            print("Warning: current_sample or scale_factors not available in _get_bbox_size. Using default.")
            if self.image_size and self.image_size[0] > 0 and self.image_size[1] > 0:
                 return int(self.image_size[0] * 0.1), int(self.image_size[1] * 0.1) # Default: 10% of image size
            return self.crop_size # Fallback: original crop_size (might be wrong scale)
        
        scale_w, scale_h = self.current_sample['scale_factors']
        crop_w, crop_h = self.crop_size
        return int(crop_w * scale_w), int(crop_h * scale_h)

    def render(self, mode='rgb_array'):
        """Render the current state of the environment"""
        if self.current_image is None:
            dummy_h = self.image_size[1] if self.image_size and len(self.image_size) > 1 else 256
            dummy_w = self.image_size[0] if self.image_size and len(self.image_size) > 0 else 256
            image_to_render = np.zeros((dummy_h, dummy_w, 3), dtype=np.uint8)
            cv2.putText(image_to_render, "No image loaded", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            image_to_render = self.current_image.copy()
        
        # Draw ground truth annotations (yellow)
        if 'annotations' in self.current_sample:
            for ann in self.current_sample['annotations']:
                bbox_ann = ann['bbox'] # Renamed
                cv2.rectangle(image_to_render, 
                              (int(bbox_ann[0]), int(bbox_ann[1])), 
                              (int(bbox_ann[0] + bbox_ann[2]), int(bbox_ann[1] + bbox_ann[3])),
                              (0, 255, 255), 1) 
        
        # Draw optimal ROIs (red)
        if self.optimal_rois:
            for roi_opt in self.optimal_rois: # Renamed
                cv2.rectangle(image_to_render,
                              (int(roi_opt[0]), int(roi_opt[1])),
                              (int(roi_opt[0] + roi_opt[2]), int(roi_opt[1] + roi_opt[3])),
                              (0, 0, 255), 1)  
                
        # Draw all placed bboxes (green)
        for bbox_placed_render in self.bboxes: # Renamed
            cv2.rectangle(image_to_render, 
                          (int(bbox_placed_render[0]), int(bbox_placed_render[1])), 
                          (int(bbox_placed_render[0] + bbox_placed_render[2]), int(bbox_placed_render[1] + bbox_placed_render[3])),
                          (0, 255, 0), 2)
        
        # Draw current movable bbox (blue)
        if self.current_bbox and len(self.current_bbox) == 4: 
            cv2.rectangle(image_to_render,
                        (int(self.current_bbox[0]), int(self.current_bbox[1])),
                        (int(self.current_bbox[0] + self.current_bbox[2]), 
                         int(self.current_bbox[1] + self.current_bbox[3])),
                        (255, 0, 0), 2)
                     
        if mode == 'rgb_array':
            return image_to_render
        elif mode == 'human':
            cv2.imshow('ROI Detection Environment', image_to_render)
            cv2.waitKey(1)
            return image_to_render 
            
    def visualize_reward_landscape(self, output_path="reward_landscape.jpg"):
        """
        Create a visualization of the reward shaping potential landscape.
        This shows the potential value for placing the *next* ROI at different locations.
        """
        if self.current_image is None or self.bbox_size is None:
            print("Cannot visualize reward landscape: current image or bbox_size is not set.")
            return None

        # Identify which optimal ROIs are already matched by currently placed bboxes
        # This is used to color-code optimal ROIs in the visualization
        matched_optimal_roi_indices_viz = [] 
        if self.optimal_rois:
            temp_matched_indices_viz = [] 
            for placed_bbox_viz_scan in self.bboxes: 
                best_iou_viz_scan = 0.0
                best_match_opt_idx_viz_scan = -1
                for i_opt_roi_viz, opt_roi_viz_item in enumerate(self.optimal_rois):
                     if i_opt_roi_viz in temp_matched_indices_viz: continue 
                     iou_viz_scan = self._calculate_iou(placed_bbox_viz_scan, opt_roi_viz_item)
                     if iou_viz_scan > best_iou_viz_scan:
                         best_iou_viz_scan = iou_viz_scan
                         best_match_opt_idx_viz_scan = i_opt_roi_viz
                if best_match_opt_idx_viz_scan != -1 and best_iou_viz_scan > 0.1:
                     if best_match_opt_idx_viz_scan not in temp_matched_indices_viz:
                          temp_matched_indices_viz.append(best_match_opt_idx_viz_scan)
            matched_optimal_roi_indices_viz = temp_matched_indices_viz

        grid_step_viz = max(1, min(self.image_size[0], self.image_size[1]) // 100) 
        img_width_viz, img_height_viz = self.image_size
        
        map_rows_viz = img_height_viz // grid_step_viz
        map_cols_viz = img_width_viz // grid_step_viz
        potential_map_viz = np.zeros((map_rows_viz, map_cols_viz))
        
        # Calculate potential for a test_bbox centered at each grid point.
        # The potential function _dist_to_nearest_unmatched_opt_roi considers self.bboxes internally.
        for r_idx_viz in range(map_rows_viz): 
            for c_idx_viz in range(map_cols_viz): 
                y_center_viz = r_idx_viz * grid_step_viz + grid_step_viz // 2 # Center of cell
                x_center_viz = c_idx_viz * grid_step_viz + grid_step_viz // 2 # Center of cell
                                 
                # Define the test bbox based on its center
                test_bbox_x = max(0, x_center_viz - self.bbox_size[0] // 2)
                test_bbox_y = max(0, y_center_viz - self.bbox_size[1] // 2)
                
                # Ensure test_bbox doesn't go out of bounds if centered near edge
                if test_bbox_x + self.bbox_size[0] > img_width_viz:
                    test_bbox_x = img_width_viz - self.bbox_size[0]
                if test_bbox_y + self.bbox_size[1] > img_height_viz:
                    test_bbox_y = img_height_viz - self.bbox_size[1]
                
                test_bbox_viz = [test_bbox_x, test_bbox_y, self.bbox_size[0], self.bbox_size[1]]
                
                # Calculate potential (negative distance, no jitter for clean visualization)
                potential_val = -self._dist_to_nearest_unmatched_opt_roi(test_bbox_viz) 
                potential_map_viz[r_idx_viz, c_idx_viz] = potential_val
        
        # Normalize potential map for visualization
        min_pot_viz = np.min(potential_map_viz)
        max_pot_viz = np.max(potential_map_viz)
        if max_pot_viz > min_pot_viz: 
            norm_potential_map_viz = (potential_map_viz - min_pot_viz) / (max_pot_viz - min_pot_viz)
        elif max_pot_viz == min_pot_viz and max_pot_viz != 0 : 
            norm_potential_map_viz = np.ones_like(potential_map_viz) * 0.5 
        else: 
            norm_potential_map_viz = np.zeros_like(potential_map_viz)
        
        plt.figure(figsize=(12, 10)) 
        plt.imshow(cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB))
        plt.imshow(norm_potential_map_viz, cmap=cm.jet, alpha=0.5, interpolation='bilinear', 
                   extent=[0, img_width_viz, img_height_viz, 0]) # y-axis inverted for images
        
        # Draw optimal ROIs (red if unmatched, green if matched by a placed bbox)
        if self.optimal_rois:
            for i_opt_render, opt_roi_render in enumerate(self.optimal_rois): 
                edge_color_opt = 'g' if i_opt_render in matched_optimal_roi_indices_viz else 'r'
                plt.gca().add_patch(plt.Rectangle(
                    (opt_roi_render[0], opt_roi_render[1]), opt_roi_render[2], opt_roi_render[3], 
                    linewidth=2, edgecolor=edge_color_opt, facecolor='none'
                ))
        
        # Draw already placed bboxes (e.g., lime green, dashed)
        for placed_bbox_render_viz in self.bboxes: 
            plt.gca().add_patch(plt.Rectangle(
                (placed_bbox_render_viz[0], placed_bbox_render_viz[1]), placed_bbox_render_viz[2], placed_bbox_render_viz[3], 
                linewidth=2, edgecolor='lime', facecolor='none', linestyle='--'
            ))
        
        # Add colorbar reflecting actual potential values before normalization
        s_map = cm.ScalarMappable(cmap=cm.jet)
        s_map.set_array([min_pot_viz, max_pot_viz]) # Set array to actual min/max for colorbar scale
        cbar = plt.colorbar(s_map, ax=plt.gca()) 
        cbar.set_label('Potential Value (Higher is Better, towards Unmatched Optimal ROI)')
        
        plt.title('Reward Shaping Potential Landscape (for next ROI placement)')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        
        legend_elements_viz = [
            Patch(facecolor='none', edgecolor='r', linewidth=2, label='Unmatched Optimal ROI'),
            Patch(facecolor='none', edgecolor='g', linewidth=2, label='Matched Optimal ROI'),
            Patch(facecolor='none', edgecolor='lime', linewidth=2, linestyle='--', label='Placed Bbox')
        ]
        plt.legend(handles=legend_elements_viz, loc='upper right', bbox_to_anchor=(1.0, 1.15)) # Adjust legend
        
        plt.tight_layout() 
        plt.savefig(output_path, dpi=300) 
        plt.close()
        
        print(f"Reward landscape visualization saved to {output_path}")
        return output_path