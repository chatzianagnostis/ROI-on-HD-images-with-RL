"""
Region of Interest Detection Environment for Reinforcement Learning.

This module implements a Gym environment for training agents to detect regions of interest
in images using reinforcement learning. The agent learns to place bounding boxes (ROIs)
strategically to cover annotations in the images.
"""

import time
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import gym
from gym import spaces
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch


class ROIDetectionEnv(gym.Env):
    """
    A Gym environment for ROI detection using reinforcement learning.
    
    The agent's task is to place bounding boxes (ROIs) on an image to optimally cover
    ground truth annotations. The agent can move a candidate ROI, place it, remove previously
    placed ROIs, or end the episode.
    
    Rewards are based on coverage of ground truth annotations and alignment with optimal ROIs
    determined by K-means clustering of annotations.
    """

    #---------------------------------------------------------------------------
    # Initialization and core environment methods (Gym interface)
    #---------------------------------------------------------------------------

    def __init__(self, dataset, crop_size: Tuple[int, int] = (640, 640), time_limit: int = 120):
        """
        Initialize the ROI Detection Environment.
        
        Args:
            dataset: ROIDataset instance for loading images and annotations
            crop_size: Size of the ROIs to place (width, height)
            time_limit: Time limit for an episode in seconds
        """
        super(ROIDetectionEnv, self).__init__()
        
        # Dataset and image information
        self.dataset = dataset
        self.image_size = dataset.image_size
        self.current_sample = None
        self.current_image = None
        
        # ROI parameters
        self.crop_size = crop_size
        self.bbox_size = None
        self.bboxes = []  # Placed bounding boxes
        self.current_bbox = None  # Current movable bounding box
        
        # Episode parameters
        self.time_limit = time_limit
        self.start_time = None
        self.optimal_rois = None  # Calculated optimal ROIs based on annotations
        
        # Define action space:
        # 0-3: Move bbox (up, down, left, right)
        # 4: Place bbox
        # 5: Remove bbox
        # 6: End episode
        self.action_space = spaces.Discrete(7)

        # Reward shaping parameters
        self.shaping_coeff = 0.00  # Coefficient for shaping reward
        self.gamma_potential_shaping = 0.995  # Discount factor for potential shaping
        
        # Observation space: image and current bbox state
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0, 
                high=255, 
                shape=(self.image_size[1], self.image_size[0], 3), 
                dtype=np.uint8
            ),
            'bbox_state': spaces.Box(
                low=0, 
                high=1, 
                shape=(100, 4),  # max 100 bboxes
                dtype=np.float32
            )
        })

    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset the environment with a new image.
        
        Returns:
            Dict: The initial observation containing the image and bbox state
        """
        # Reset episode timer
        self.start_time = time.time()
        
        # Get a new sample from the dataset
        try:
            self.current_sample = next(self.dataset)
        except StopIteration:
            # Dataset exhausted — restart it
            self.dataset.__iter__()  # Resets index and optionally reshuffles
            self.current_sample = next(self.dataset)

        # Get the resized image from the sample
        self.current_image = self.current_sample['resized_image']

        # Calculate visual bbox size for the resized image
        self.bbox_size = self._get_bbox_size()
        
        # Process annotations that are taller than bbox size
        self._process_tall_annotations()
        
        # Calculate optimal ROIs using K-means
        self.optimal_rois = self._calculate_optimal_rois()

        # Reset list of placed bounding boxes
        self.bboxes = []

        # Initialize the current movable bounding box at the center
        if self.bbox_size is None or not (len(self.bbox_size) == 2): 
            # Fallback if bbox_size is somehow not set
            print("Warning: bbox_size is None or invalid in reset. Using default values.")
            self.current_bbox = [
                0, 0, 
                self.crop_size[0] // 10, 
                self.crop_size[1] // 10
            ]  # Arbitrary small default
        else:
            self.current_bbox = [
                (self.image_size[0] - self.bbox_size[0]) // 2,  # Center x
                (self.image_size[1] - self.bbox_size[1]) // 2,  # Center y
                self.bbox_size[0],  # Width
                self.bbox_size[1]   # Height
            ]
            
        # Return the initial observation
        return self._get_observation()

    def step(self, action_from_agent: int) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """
        Take a step in the environment based on the agent's action.
        
        Args:
            action_from_agent: The action selected by the agent (0-6)
            
        Returns:
            Tuple containing:
                - observation: Dict with the new state
                - reward: Float value of the reward
                - done: Boolean indicating if the episode is finished
                - info: Dict with additional information
        """
        # Initialize reward, done flag, and info dictionary
        current_reward_for_action = 0.0
        done = False
        info = {}
        metrics = {}

        # Check if time limit has been reached
        elapsed_time = time.time() - self.start_time

        if elapsed_time > self.time_limit:
            # Time limit reached - end episode without executing action
            done = True
            info['TimeLimit.truncated'] = True
            print(f"Time limit ({self.time_limit}s) reached. "
                  f"Agent's action {action_from_agent} was not executed.")
        else:
            # Time limit not reached - proceed with action execution
            info['TimeLimit.truncated'] = False

            # Execute the selected action (all actions are valid)
            if action_from_agent < 4:  # Move bbox (0: Up, 1: Down, 2: Left, 3: Right)
                if self.current_bbox:
                    old_potential = self._potential_function(self.current_bbox)
                    self._move_bbox(action_from_agent)
                    new_potential = self._potential_function(self.current_bbox)
                    shaping_reward = self.gamma_potential_shaping * new_potential - old_potential
                    current_reward_for_action = shaping_reward * self.shaping_coeff
                else:
                    print("Warning: Agent tried to move a None or invalid current_bbox.")
                    current_reward_for_action = -1  # Penalty for invalid action
                    
            elif action_from_agent == 4:  # Place bbox
                current_reward_for_action = self._place_bbox()
                
            elif action_from_agent == 5:  # Remove bbox
                current_reward_for_action = self._remove_bbox()
                
            elif action_from_agent == 6:  # End episode
                done = True

        # Calculate total reward for this step
        total_step_reward = current_reward_for_action

        # Add final reward if episode is ending
        if done:
            try:
                final_reward_value, metrics_from_final_reward = self._calculate_final_reward()
                total_step_reward += final_reward_value
                
                # Update metrics
                if metrics_from_final_reward:
                    metrics.update(metrics_from_final_reward)
                metrics['time_elapsed_at_done'] = elapsed_time
                metrics['time_limit_episode'] = self.time_limit

            except Exception as e:
                print(f"Error during final reward calculation: {e}")
            
            # Add metrics to info dictionary
            info['metrics'] = metrics if metrics else {}

        # Get observation for next state
        try:
            observation = self._get_observation()
        except Exception as e:
            print(f"Error getting observation: {e}")
            observation = self.observation_space.sample()  # Fallback

        return observation, total_step_reward, done, info

    def render(self, mode: str = 'rgb_array') -> np.ndarray:
        """
        Render the current state of the environment.
        
        Args:
            mode: Rendering mode ('rgb_array' or 'human')
            
        Returns:
            np.ndarray: The rendered image
        """
        # Create base image to render on
        if self.current_image is None:
            dummy_h = self.image_size[1] if self.image_size and len(self.image_size) > 1 else 256
            dummy_w = self.image_size[0] if self.image_size and len(self.image_size) > 0 else 256
            image_to_render = np.zeros((dummy_h, dummy_w, 3), dtype=np.uint8)
            cv2.putText(
                image_to_render, 
                "No image loaded", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (255, 255, 255), 
                2
            )
        else:
            image_to_render = self.current_image.copy()
        
        # Draw ground truth annotations (yellow)
        if 'annotations' in self.current_sample:
            for ann in self.current_sample['annotations']:
                bbox_ann = ann['bbox']
                cv2.rectangle(
                    image_to_render, 
                    (int(bbox_ann[0]), int(bbox_ann[1])), 
                    (int(bbox_ann[0] + bbox_ann[2]), int(bbox_ann[1] + bbox_ann[3])),
                    (0, 255, 255),  # Yellow color
                    1  # Line thickness
                )
        
        # Draw optimal ROIs (red)
        if self.optimal_rois:
            for roi_opt in self.optimal_rois:
                cv2.rectangle(
                    image_to_render,
                    (int(roi_opt[0]), int(roi_opt[1])),
                    (int(roi_opt[0] + roi_opt[2]), int(roi_opt[1] + roi_opt[3])),
                    (0, 0, 255),  # Red color
                    1  # Line thickness
                )
                
        # Draw all placed bboxes (green)
        for bbox_placed_render in self.bboxes:
            cv2.rectangle(
                image_to_render, 
                (int(bbox_placed_render[0]), int(bbox_placed_render[1])), 
                (int(bbox_placed_render[0] + bbox_placed_render[2]), 
                 int(bbox_placed_render[1] + bbox_placed_render[3])),
                (0, 255, 0),  # Green color
                2  # Line thickness
            )
        
        # Draw current movable bbox (blue)
        if self.current_bbox and len(self.current_bbox) == 4: 
            cv2.rectangle(
                image_to_render,
                (int(self.current_bbox[0]), int(self.current_bbox[1])),
                (int(self.current_bbox[0] + self.current_bbox[2]), 
                 int(self.current_bbox[1] + self.current_bbox[3])),
                (255, 0, 0),  # Blue color
                2  # Line thickness
            )
                     
        if mode == 'rgb_array':
            return image_to_render
        elif mode == 'human':
            cv2.imshow('ROI Detection Environment', image_to_render)
            cv2.waitKey(1)
            return image_to_render

    # Keep the action_masks method for compatibility
    def action_masks(self) -> np.ndarray:
        """
        Return the valid action mask for the current state.
        
        This method is kept for compatibility but returns all ones since
        we're treating all actions as valid.
        
        Returns:
            np.ndarray: Binary mask with all 1s (all actions allowed)
        """
        return np.ones(self.action_space.n, dtype=np.int8)

    #---------------------------------------------------------------------------
    # Observation and action processing methods
    #---------------------------------------------------------------------------
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Get the current observation.
        
        Returns:
            Dict containing:
                - 'image': The current image with all bounding boxes drawn
                - 'bbox_state': Normalized coordinates of all placed bboxes
        """
        # Create a copy of the current image to draw on
        image = self.current_image.copy()
        
        # Draw all placed bboxes in green
        for bbox in self.bboxes:
            cv2.rectangle(
                image, 
                (int(bbox[0]), int(bbox[1])), 
                (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                (0, 255, 0),  # Green color
                2  # Line thickness
            )
        
        # Draw current movable bbox in blue
        if self.current_bbox and len(self.current_bbox) == 4:
            cv2.rectangle(
                image,
                (int(self.current_bbox[0]), int(self.current_bbox[1])),
                (int(self.current_bbox[0] + self.current_bbox[2]), 
                 int(self.current_bbox[1] + self.current_bbox[3])),
                (255, 0, 0),  # Blue color
                2  # Line thickness
            )
        
        # Prepare the bbox_state array (normalized coordinates of placed bboxes)
        bbox_state = np.zeros((100, 4), dtype=np.float32)
        for i, bbox in enumerate(self.bboxes):
            if i < 100:  # Max 100 bboxes in the state
                # Normalize coordinates to [0, 1] range
                bbox_state[i] = [
                    bbox[0] / self.image_size[0],  # x / width
                    bbox[1] / self.image_size[1],  # y / height
                    bbox[2] / self.image_size[0],  # w / width
                    bbox[3] / self.image_size[1]   # h / height
                ]
        
        return {
            'image': image.astype(np.uint8),
            'bbox_state': bbox_state
        }

    def _get_action_mask(self) -> np.ndarray:
        """
        Compute the action mask for the current state.
        
        Since we're treating all actions as valid, this returns all ones.
        Kept for compatibility with original code.
        
        Returns:
            np.ndarray: Binary mask with all 1s (all actions allowed)
        """
        # Return all actions as valid
        return np.ones(self.action_space.n, dtype=np.int8)

    def _process_tall_annotations(self) -> None:
        """
        Process annotations that are taller than the bbox size by cropping them.
        
        This ensures that all annotations can be potentially covered by
        the agent's bounding boxes.
        """
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
            
            # If annotation height is within bbox height, keep it unchanged
            if bbox[3] <= self.bbox_size[1]:
                processed_annotations.append(ann)
                continue
            
            # If annotation is too tall, crop it to bbox height
            top_part = ann.copy()
            top_part['bbox'] = [bbox[0], bbox[1], bbox[2], self.bbox_size[1]]
            processed_annotations.append(top_part)
            
        self.current_sample['annotations'] = processed_annotations

    def _move_bbox(self, direction: int) -> float:
        """
        Move the current bounding box in the specified direction.
        Returns a penalty if trying to move beyond image boundaries.
        
        Args:
            direction: Direction to move (0: Up, 1: Down, 2: Left, 3: Right)
            
        Returns:
            float: Reward/penalty (0 for valid move, -1 for invalid move)
        """
        # Define step size for movement
        step_size = 8
        
        # Validate bbox_size and current_bbox before moving
        if (self.bbox_size is None or len(self.bbox_size) != 2 or 
            self.current_bbox is None or len(self.current_bbox) != 4):
            print("Warning: bbox_size or current_bbox is invalid in _move_bbox.")
            return -1.0
        
        # Check if movement would go beyond boundaries and apply penalty
        if direction == 0:  # Up
            if self.current_bbox[1] <= 0:
                return -1.0  # Already at top edge, can't move up
            self.current_bbox[1] = max(0, self.current_bbox[1] - step_size)
                
        elif direction == 1:  # Down
            bottom_edge = self.image_size[1] - self.bbox_size[1]
            if self.current_bbox[1] >= bottom_edge:
                return -1.0  # Already at bottom edge, can't move down
            self.current_bbox[1] = min(bottom_edge, self.current_bbox[1] + step_size)
                
        elif direction == 2:  # Left
            if self.current_bbox[0] <= 0:
                return -1.0  # Already at left edge, can't move left
            self.current_bbox[0] = max(0, self.current_bbox[0] - step_size)
                
        elif direction == 3:  # Right
            right_edge = self.image_size[0] - self.bbox_size[0]
            if self.current_bbox[0] >= right_edge:
                return -1.0  # Already at right edge, can't move right
            self.current_bbox[0] = min(right_edge, self.current_bbox[0] + step_size)
        
        # If we reach here, the move was valid
        return 0.0


    def _place_bbox(self) -> float:
        """
        Place the current bounding box and calculate the resulting reward.
        
        Logic:
        1. Check if we've already placed 99 boxes (limit) - if so, return penalty
        2. Place the box
        3. Check for high overlap with existing boxes - if found, return penalty
        4. Check IoU with optimal ROIs and return highest IoU * 10 as reward
        
        Returns:
            float: The reward for this action
        """
        # 1. Check if already at max boxes (99)
        if len(self.bboxes) >= 99:
            return -1.0  # Penalty for exceeding limit
        
        # Create a copy of the current bbox to place
        current_bbox_to_place = self.current_bbox.copy()
        
        # 2. Place the bbox (add to list)
        self.bboxes.append(current_bbox_to_place)
        
        # 3. Check for high overlap with existing boxes (except the one we just placed)
        for existing_bbox in self.bboxes[:-1]:  # Skip the last one (just placed)
            iou_with_existing = self._calculate_iou(current_bbox_to_place, existing_bbox)
            if iou_with_existing > 0.9:
                return -1.0  # Penalty for high overlap
        
        # 4. Check IoU with each optimal ROI and return highest * 10 as reward
        max_iou_with_optimal = 0.0
        
        if self.optimal_rois:
            for opt_roi in self.optimal_rois:
                iou_val = self._calculate_iou(current_bbox_to_place, opt_roi)
                max_iou_with_optimal = max(max_iou_with_optimal, iou_val)
        
        # Return reward as highest IoU with optimal ROI * 10
        return max_iou_with_optimal * 10.0

    def _remove_bbox(self) -> float:
        """
        Remove the last placed bounding box and calculate the resulting reward.
        
        Returns:
            float: The reward for this action
        """
        # Check if there are any boxes to remove
        if not self.bboxes:
            return -1  # Penalty if nothing to remove
            
        # Remove the box
        self.bboxes.pop()
        return 0.0

    #---------------------------------------------------------------------------
    # ROI and reward calculation methods
    #---------------------------------------------------------------------------

    def _calculate_optimal_rois(self) -> List[List[float]]:
        """
        Calculate optimal ROIs using K-means clustering on annotation centers.
        
        The method finds cluster centers and creates ROIs of the specified bbox_size
        around these centers, ensuring all annotations in a cluster can be covered
        by a single ROI.
        
        Returns:
            List of ROIs [x, y, w, h]
        """
        annotations = self.current_sample.get('annotations', [])
        
        # Validate bbox_size
        if self.bbox_size is None or len(self.bbox_size) != 2:
             print("Critical Warning: bbox_size not properly initialized in _calculate_optimal_rois.")
             self.bbox_size = self._get_bbox_size()  # Try to recover
             if self.bbox_size is None:
                 self.bbox_size = (self.crop_size[0]*0.1, self.crop_size[1]*0.1)  # Fallback

        roi_w, roi_h = self.bbox_size
        half_w, half_h = roi_w / 2.0, roi_h / 2.0
        H, W = self.current_image.shape[:2]

        # If no annotations, create a single ROI in the center
        if not annotations:
            cx, cy = W / 2.0, H / 2.0
            x0 = max(0, cx - half_w)
            y0 = max(0, cy - half_h)
            clamped_w = min(roi_w, W - x0)
            clamped_h = min(roi_h, H - y0)
            return [[x0, y0, clamped_w, clamped_h]]

        # Helper function to clamp center coordinates
        def clamp_center(val, half_dim, total_dim):
            return max(half_dim, min(val, total_dim - half_dim))
            
        # Collect annotation centers
        pts = []
        for ann in annotations:
            x, y, w_ann, h_ann = ann['bbox']
            pts.append((x + w_ann / 2.0, y + h_ann / 2.0))
        
        # Handle edge cases
        if not pts:
             cx, cy = W/2.0, H/2.0
             x0 = max(0, cx - half_w)
             y0 = max(0, cy - half_h)
             return [[x0, y0, min(roi_w, W-x0), min(roi_h, H-y0)]]
             
        if len(pts) == 1:
            cx, cy = pts[0]
            cx = clamp_center(cx, half_w, W)
            cy = clamp_center(cy, half_h, H)
            x0 = cx - half_w
            y0 = cy - half_h
            return [[x0, y0, roi_w, roi_h]]
            
        # Convert to numpy array for K-means
        pts_np = np.array(pts)
        
        # Start with each point as its own cluster
        best_k = len(pts_np)
        best_centers = pts_np

        # Try different k values to find minimum k that satisfies constraints
        for k_val in range(1, len(pts_np) + 1):
            if len(pts_np) < k_val:
                continue
            
            try:
                km = KMeans(n_clusters=k_val, random_state=0, n_init='auto').fit(pts_np)
            except ValueError:
                if k_val == 1 and len(pts_np) >= 1:
                     best_centers = np.mean(pts_np, axis=0, keepdims=True)
                     best_k = 1
                     break
                continue

            current_centers = km.cluster_centers_
            labels = km.labels_
            
            # Check if this k satisfies the L-infinity constraint
            is_good_k = True
            for cluster_idx in range(k_val):
                cluster_center_x, cluster_center_y = current_centers[cluster_idx]
                member_points = pts_np[labels == cluster_idx]
                
                if not member_points.any():
                    continue
                
                # Calculate L-infinity radius for this cluster
                max_delta_x = np.max(np.abs(member_points[:, 0] - cluster_center_x))
                max_delta_y = np.max(np.abs(member_points[:, 1] - cluster_center_y))

                # Check if all points fit within ROI centered at cluster center
                if max_delta_x > half_w or max_delta_y > half_h:
                    is_good_k = False
                    break
            
            if is_good_k:
                best_k = k_val
                best_centers = current_centers
                break

        # Create ROIs from cluster centers
        final_rois = []
        if best_centers is not None and len(best_centers) > 0:
            for (center_x, center_y) in best_centers:
                # Clamp centers to ensure ROIs are within image bounds
                clamped_center_x = clamp_center(center_x, half_w, W)
                clamped_center_y = clamp_center(center_y, half_h, H)
                
                x0 = max(0, clamped_center_x - half_w)
                y0 = max(0, clamped_center_y - half_h)

                final_rois.append([x0, y0, roi_w, roi_h])
        else:
             # Fallback to center of image
             cx_fb, cy_fb = W/2.0, H/2.0
             x0_fb = max(0, cx_fb - half_w)
             y0_fb = max(0, cy_fb - half_h)
             final_rois.append([x0_fb, y0_fb, min(roi_w, W-x0_fb), min(roi_h, H-y0_fb)])
             
        return final_rois

    def _dist_to_nearest_unmatched_opt_roi(self, bbox: List[float]) -> float:
        """
        Calculate the minimum Manhattan distance from the center of the current bbox
        to the center of any optimal ROI that hasn't been matched yet by a placed bbox.
        
        This is used for reward shaping to guide the agent toward uncovered optimal ROIs.
        
        Args:
            bbox: The bounding box to calculate distance from [x, y, w, h]
            
        Returns:
            float: Minimum distance to nearest unmatched optimal ROI, or 0 if none exist
        """
        if not self.optimal_rois:
            return 0.0
        
        # Calculate center of the input bbox
        bbox_cx = bbox[0] + bbox[2] / 2.0
        bbox_cy = bbox[1] + bbox[3] / 2.0
        
        # Identify indices of optimal ROIs already "matched" by placed bboxes
        matched_opt_rois_indices = [] 
        temp_optimal_rois_for_matching = list(enumerate(self.optimal_rois))

        # Find which optimal ROIs are already covered by placed bboxes
        for placed_bbox_in_list in self.bboxes:
            best_iou_for_placed = 0.0
            best_match_opt_idx = -1
            
            for i, opt_roi_data in temp_optimal_rois_for_matching:
                if i in matched_opt_rois_indices:  # Skip already matched ROIs
                    continue
                    
                iou = self._calculate_iou(placed_bbox_in_list, opt_roi_data)
                if iou > best_iou_for_placed:
                    best_iou_for_placed = iou
                    best_match_opt_idx = i
            
            # If a good match is found, mark this optimal ROI as matched
            if best_match_opt_idx != -1 and best_iou_for_placed > 0.1:
                if best_match_opt_idx not in matched_opt_rois_indices:
                    matched_opt_rois_indices.append(best_match_opt_idx)
        
        # Calculate minimum distance to any unmatched optimal ROI
        min_dist = float('inf')
        unmatched_roi_exists = False
        
        for i, roi in enumerate(self.optimal_rois):
            if i in matched_opt_rois_indices:
                continue  # Skip matched ROIs
                
            unmatched_roi_exists = True
            
            # Calculate center of the optimal ROI
            roi_cx = roi[0] + roi[2] / 2.0
            roi_cy = roi[1] + roi[3] / 2.0
            
            # Use Manhattan distance (L1 norm)
            dist = abs(bbox_cx - roi_cx) + abs(bbox_cy - roi_cy)
            min_dist = min(min_dist, dist)
        
        if not unmatched_roi_exists:
            return 0.0  # All optimal ROIs are matched
        
        return min_dist if min_dist != float('inf') else 0.0

    def _potential_function(self, bbox: List[float]) -> float:
        """
        Potential function for reward shaping.
        
        Returns a higher value when closer to an unmatched optimal ROI.
        Includes small random jitter to prevent getting stuck in local optima.
        
        Args:
            bbox: The bounding box to evaluate [x, y, w, h]
            
        Returns:
            float: Potential value (higher is better)
        """
        # Get distance to nearest unmatched optimal ROI
        distance = self._dist_to_nearest_unmatched_opt_roi(bbox)
        
        # Generate seeded random jitter based on bbox position for consistency
        jitter_seed = int(bbox[0] * 1000 + bbox[1])  # Simple hash of position
        jitter_seed = abs(int(jitter_seed)) % (2**32)  # Ensure positive seed within limits
        rng = np.random.RandomState(jitter_seed) 
        
        # Scale jitter proportionally to image size but keep it small
        jitter_scale = min(self.image_size[0], self.image_size[1]) * 0.005
        jitter = rng.normal(0, jitter_scale)
        
        # Potential is higher (less negative) when distance is smaller
        return -distance + jitter

    def _calculate_final_reward(self) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate final reward based on annotation coverage and ROI matching.
        
        Returns:
            Tuple containing:
                - float: The final reward value
                - Dict: Metrics about the episode performance
        """
        # Check if any ROIs were placed
        if not self.bboxes:
            return -20.0, {"metrics": "No ROIs placed"}

        # Ensure optimal_rois is calculated
        if self.optimal_rois is None:
            self.optimal_rois = self._calculate_optimal_rois()
        
        # Get annotations and calculate scores
        annotations = self.current_sample.get('annotations', [])
        coverage_score = self._calculate_annotation_coverage(self.bboxes, annotations)
        roi_matching_score = self._calculate_roi_matching(self.bboxes, self.optimal_rois)
        
        # Count optimal and placed ROIs
        optimal_count = len(self.optimal_rois) if self.optimal_rois else 0
        placed_count = len(self.bboxes)

        # Calculate final reward with weights for different components
        coverage_weight = 70.0
        matching_weight = 30.0
        
        final_reward_val = (
            coverage_score * coverage_weight +
            roi_matching_score * matching_weight
        )
        
        # Collect metrics for logging
        calculated_metrics = {
            'optimal_rois_list': self.optimal_rois,
            'coverage_score': coverage_score,
            'roi_matching_score': roi_matching_score,
            'optimal_count': optimal_count,
            'placed_count': placed_count
        }
        
        return final_reward_val, calculated_metrics
    
    def _calculate_annotation_coverage(self, placed_rois: List[List[float]], 
                                       annotations: List[Dict]) -> float:
        """
        Calculate the fraction of annotations properly covered by placed ROIs.
        
        Args:
            placed_rois: List of placed ROIs [x, y, w, h]
            annotations: List of annotation dictionaries
            
        Returns:
            float: Coverage score between 0 and 1
        """
        if not annotations:
            return 1.0  # Perfect coverage if no annotations to cover
        
        covered_count = 0
        total_annotations = len(annotations)
        
        for ann in annotations:
            ann_bbox = ann['bbox']
            for roi in placed_rois:
                if self._is_bbox_contained(ann_bbox, roi):
                    covered_count += 1
                    break  # This annotation is covered, move to the next
        
        return covered_count / total_annotations if total_annotations > 0 else 1.0

    def _calculate_roi_matching(self, placed_rois: List[List[float]], 
                               optimal_rois_list: List[List[float]]) -> float:
        """
        Calculate how well placed ROIs match optimal ROIs using IoU.
        
        Args:
            placed_rois: List of placed ROIs [x, y, w, h]
            optimal_rois_list: List of optimal ROIs [x, y, w, h]
            
        Returns:
            float: Matching score between 0 and 1
        """
        if not optimal_rois_list or not placed_rois:
            return 0.0
        
        total_max_iou_sum = 0.0
        
        # For each optimal ROI, find the placed ROI with highest IoU
        for opt_roi in optimal_rois_list:
            best_iou_for_this_optimal_roi = 0.0
            for p_roi in placed_rois:
                iou = self._calculate_iou(opt_roi, p_roi)
                best_iou_for_this_optimal_roi = max(best_iou_for_this_optimal_roi, iou)
            total_max_iou_sum += best_iou_for_this_optimal_roi
        
        # Average the best IoUs
        return total_max_iou_sum / len(optimal_rois_list)

    #---------------------------------------------------------------------------
    # Utility methods
    #---------------------------------------------------------------------------
    
    def _get_bbox_size(self) -> Tuple[int, int]:
        """
        Calculate the bbox size in the resized image.
        
        Returns:
            Tuple containing width and height of the bbox
        """
        if self.current_sample is None or 'scale_factors' not in self.current_sample:
            print("Warning: current_sample or scale_factors not available in _get_bbox_size.")
            if self.image_size and self.image_size[0] > 0 and self.image_size[1] > 0:
                 return int(self.image_size[0] * 0.1), int(self.image_size[1] * 0.1)
            return self.crop_size
        
        scale_w, scale_h = self.current_sample['scale_factors']
        crop_w, crop_h = self.crop_size
        return int(crop_w * scale_w), int(crop_h * scale_h)

    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1: First bounding box [x, y, w, h]
            box2: Second bounding box [x, y, w, h]
            
        Returns:
            float: IoU value between 0 and 1
        """
        # Convert [x, y, w, h] to [x1, y1, x2, y2] format
        box1_x1, box1_y1 = box1[0], box1[1]
        box1_x2, box1_y2 = box1[0] + box1[2], box1[1] + box1[3]
        
        box2_x1, box2_y1 = box2[0], box2[1]
        box2_x2, box2_y2 = box2[0] + box2[2], box2[1] + box2[3]
        
        # Calculate intersection coordinates
        x_left = max(box1_x1, box2_x1)
        y_top = max(box1_y1, box2_y1)
        x_right = min(box1_x2, box2_x2)
        y_bottom = min(box1_y2, box2_y2)
        
        # Check if there is any overlap
        if x_right < x_left or y_bottom < y_top:
            return 0.0  # No overlap
            
        # Calculate areas
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = box1[2] * box1[3]  # w * h
        box2_area = box2[2] * box2[3]  # w * h
        union_area = box1_area + box2_area - intersection_area
        
        # Avoid division by zero
        if union_area == 0:
            return 0.0
        
        # Calculate IoU
        iou = intersection_area / union_area
        return iou

    def _is_bbox_contained(self, bbox1_ann: List[float], bbox2_roi: List[float], 
                          threshold: float = 0.8) -> bool:
        """
        Check if bbox1_ann (annotation) is mostly contained within bbox2_roi (ROI).
        
        Args:
            bbox1_ann: Annotation bounding box [x, y, w, h]
            bbox2_roi: ROI bounding box [x, y, w, h]
            threshold: Minimum containment ratio (0.8 means 80% contained)
            
        Returns:
            bool: True if annotation is mostly contained within ROI
        """
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
        
        if intersection_area == 0:
            return False
                
        bbox1_area = b1_w * b1_h
        
        if bbox1_area == 0:
             return False
             
        # Return true if intersection covers at least threshold of annotation's area
        return (intersection_area / bbox1_area) >= threshold

    def _scale_bbox_to_original(self, bbox: List[float]) -> List[int]:
        """
        Scale a bbox from the resized image back to the original image dimensions.
        
        Args:
            bbox: Bounding box in resized image coordinates [x, y, w, h]
            
        Returns:
            List[int]: Bounding box in original image coordinates [x, y, w, h]
        """
        if 'scale_factors' not in self.current_sample:
            return bbox  # Return unchanged if scale factors not available
            
        scale_factors = self.current_sample['scale_factors']
        
        # Apply inverse scaling to convert back to original dimensions
        original_bbox = [
            int(bbox[0] / scale_factors[0]),  # x
            int(bbox[1] / scale_factors[1]),  # y
            int(bbox[2] / scale_factors[0]),  # width
            int(bbox[3] / scale_factors[1])   # height
        ]
        
        return original_bbox

    def visualize_reward_landscape(self, output_path: str = "reward_landscape.jpg") -> Optional[str]:
        """
        Create a visualization of the reward shaping potential landscape.
        
        This shows the potential value for placing the next ROI at different locations,
        which helps understand where the agent is incentivized to place boxes.
        
        Args:
            output_path: File path to save the visualization
            
        Returns:
            str: Path to the saved visualization, or None if visualization failed
        """
        # Check if current state is valid for visualization
        if self.current_image is None or self.bbox_size is None:
            print("Cannot visualize reward landscape: current image or bbox_size is not set.")
            return None

        # Identify which optimal ROIs are already matched by currently placed bboxes
        matched_optimal_roi_indices = []
        if self.optimal_rois:
            # Find which optimal ROIs are matched by placed bboxes
            for placed_bbox in self.bboxes:
                best_iou = 0.0
                best_match_idx = -1
                
                for i, opt_roi in enumerate(self.optimal_rois):
                    # Skip already matched ROIs
                    if i in matched_optimal_roi_indices:
                        continue
                        
                    iou = self._calculate_iou(placed_bbox, opt_roi)
                    if iou > best_iou:
                        best_iou = iou
                        best_match_idx = i
                        
                # If found a good match, add to matched indices
                if best_match_idx != -1 and best_iou > 0.1:
                    if best_match_idx not in matched_optimal_roi_indices:
                        matched_optimal_roi_indices.append(best_match_idx)

        # Set up grid for evaluating potential
        grid_step = max(1, min(self.image_size[0], self.image_size[1]) // 100)
        img_width, img_height = self.image_size
        
        map_rows = img_height // grid_step
        map_cols = img_width // grid_step
        potential_map = np.zeros((map_rows, map_cols))
        
        # Calculate potential for each grid point
        for r_idx in range(map_rows):
            for c_idx in range(map_cols):
                # Calculate center coordinates of this grid cell
                y_center = r_idx * grid_step + grid_step // 2
                x_center = c_idx * grid_step + grid_step // 2
                
                # Define a test bbox centered at this point
                test_bbox_x = max(0, x_center - self.bbox_size[0] // 2)
                test_bbox_y = max(0, y_center - self.bbox_size[1] // 2)
                
                # Ensure test_bbox stays within image boundaries
                if test_bbox_x + self.bbox_size[0] > img_width:
                    test_bbox_x = img_width - self.bbox_size[0]
                if test_bbox_y + self.bbox_size[1] > img_height:
                    test_bbox_y = img_height - self.bbox_size[1]
                
                test_bbox = [test_bbox_x, test_bbox_y, self.bbox_size[0], self.bbox_size[1]]
                
                # Calculate potential (negative distance without jitter for visualization)
                potential_val = -self._dist_to_nearest_unmatched_opt_roi(test_bbox)
                potential_map[r_idx, c_idx] = potential_val
        
        # Normalize potential map for visualization
        min_pot = np.min(potential_map)
        max_pot = np.max(potential_map)
        
        if max_pot > min_pot:
            norm_potential_map = (potential_map - min_pot) / (max_pot - min_pot)
        elif max_pot == min_pot and max_pot != 0:
            norm_potential_map = np.ones_like(potential_map) * 0.5
        else:
            norm_potential_map = np.zeros_like(potential_map)
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        plt.imshow(cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB))
        plt.imshow(
            norm_potential_map,
            cmap=cm.jet,
            alpha=0.5,
            interpolation='bilinear',
            extent=[0, img_width, img_height, 0]  # y-axis inverted for images
        )
        
        # Draw optimal ROIs (red if unmatched, green if matched)
        if self.optimal_rois:
            for i, opt_roi in enumerate(self.optimal_rois):
                edge_color = 'g' if i in matched_optimal_roi_indices else 'r'
                plt.gca().add_patch(plt.Rectangle(
                    (opt_roi[0], opt_roi[1]),
                    opt_roi[2], opt_roi[3],
                    linewidth=2,
                    edgecolor=edge_color,
                    facecolor='none'
                ))
        
        # Draw already placed bboxes
        for placed_bbox in self.bboxes:
            plt.gca().add_patch(plt.Rectangle(
                (placed_bbox[0], placed_bbox[1]),
                placed_bbox[2], placed_bbox[3],
                linewidth=2,
                edgecolor='lime',
                facecolor='none',
                linestyle='--'
            ))
        
        # Add colorbar reflecting actual potential values
        s_map = cm.ScalarMappable(cmap=cm.jet)
        s_map.set_array([min_pot, max_pot])  # Set actual min/max for colorbar scale
        cbar = plt.colorbar(s_map, ax=plt.gca())
        cbar.set_label('Potential Value (Higher is Better, towards Unmatched Optimal ROI)')
        
        # Add title and axes labels
        plt.title('Reward Shaping Potential Landscape (for next ROI placement)')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        
        # Add legend
        legend_elements = [
            Patch(facecolor='none', edgecolor='r', linewidth=2, label='Unmatched Optimal ROI'),
            Patch(facecolor='none', edgecolor='g', linewidth=2, label='Matched Optimal ROI'),
            Patch(facecolor='none', edgecolor='lime', linewidth=2, linestyle='--', label='Placed Bbox')
        ]
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.15))
        
        # Save and close
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"Reward landscape visualization saved to {output_path}")
        return output_path