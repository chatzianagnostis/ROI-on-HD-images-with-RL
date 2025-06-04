"""
Region of Interest Detection Environment for Reinforcement Learning.

This module implements a Gym environment for training agents to detect regions of interest
in images using reinforcement learning. The agent learns to place bounding boxes (ROIs)
to cover annotations in the images.

Enhanced with action history, position history, and movement features - NO reward changes!
"""

import time
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from collections import deque

import numpy as np
import gym
from gym import spaces
import cv2


class ROIDetectionEnv(gym.Env):
    """
    A Gym environment for ROI detection using reinforcement learning.
    
    The agent's task is to place bounding boxes (ROIs) on an image to cover
    ground truth annotations. The agent can move a candidate ROI, place it, 
    remove previously placed ROIs, or end the episode.
    
    Rewards are based on covering annotations efficiently with IoU > 0.5.
    """

    def __init__(self, dataset, crop_size: Tuple[int, int] = (640, 640), 
                 time_limit: int = 120, iou_threshold: float = 0.5,
                 action_history_length: int = 10, track_positions: bool = True):
        """
        Initialize the ROI Detection Environment.
        
        Args:
            dataset: ROIDataset instance for loading images and annotations
            crop_size: Size of the ROIs to place (width, height)
            time_limit: Time limit for an episode in seconds
            iou_threshold: IoU threshold for considering an annotation "covered"
            action_history_length: Number of recent actions to remember
            track_positions: Whether to track recent bbox positions
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
        
        # Reward parameters
        self.iou_threshold = iou_threshold
        self.covered_annotations = set()  # Track which annotations are already covered
        
        # NEW: Action and position history tracking
        self.action_history_length = action_history_length
        self.track_positions = track_positions
        self.action_history = deque(maxlen=action_history_length)
        self.position_history = deque(maxlen=action_history_length) if track_positions else None
        
        # Define action space:
        # 0-3: Move bbox (up, down, left, right)
        # 4: Place bbox
        # 5: Remove bbox
        # 6: End episode
        self.action_space = spaces.Discrete(7)
        
        # Enhanced observation space with history features
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
            ),
            'current_bbox': spaces.Box(
                low=0, 
                high=1, 
                shape=(4,), 
                dtype=np.float32
            ),
            # NEW: Action history
            'action_history': spaces.Box(
                low=0, high=7,  # Actions 0-6 + padding token 7
                shape=(action_history_length,),
                dtype=np.int32
            ),
            # NEW: Position history
            'position_history': spaces.Box(
                low=0, high=1,
                shape=(action_history_length, 2) if track_positions else (1,),
                dtype=np.float32
            ),
            # NEW: Movement pattern features
            'movement_features': spaces.Box(
                low=-1, high=1,
                shape=(6,),  # Various movement statistics
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

        # Reset list of placed bounding boxes
        self.bboxes = []
        
        # Reset covered annotations tracker
        self.covered_annotations = set()

        # NEW: Clear action and position history
        self.action_history.clear()
        if self.position_history is not None:
            self.position_history.clear()
        
        # Initialize history with placeholder values
        for _ in range(self.action_history_length):
            self.action_history.append(-1)  # -1 = no action yet
            if self.position_history is not None:
                self.position_history.append([0.5, 0.5])  # Center position

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
        # NEW: Store current position before action
        if self.position_history is not None:
            current_pos = [
                self.current_bbox[0] / self.image_size[0],
                self.current_bbox[1] / self.image_size[1]
            ]
            self.position_history.append(current_pos)
        
        # NEW: Add action to history
        self.action_history.append(action_from_agent)
        
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

            # Execute the selected action (ORIGINAL REWARD LOGIC - NO CHANGES)
            if action_from_agent < 4:  # Move bbox (0: Up, 1: Down, 2: Left, 3: Right)
                self._move_bbox(action_from_agent)
                current_reward_for_action = 0.0  # No reward for moving
                
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
        
        # Draw ground truth annotations (yellow for uncovered, green for covered)
        if 'annotations' in self.current_sample:
            for i, ann in enumerate(self.current_sample['annotations']):
                bbox_ann = ann['bbox']
                color = (0, 255, 0) if i in self.covered_annotations else (0, 255, 255)
                cv2.rectangle(
                    image_to_render, 
                    (int(bbox_ann[0]), int(bbox_ann[1])), 
                    (int(bbox_ann[0] + bbox_ann[2]), int(bbox_ann[1] + bbox_ann[3])),
                    color,
                    1  # Line thickness
                )
                
        # Draw all placed bboxes (blue)
        for bbox_placed_render in self.bboxes:
            cv2.rectangle(
                image_to_render, 
                (int(bbox_placed_render[0]), int(bbox_placed_render[1])), 
                (int(bbox_placed_render[0] + bbox_placed_render[2]), 
                 int(bbox_placed_render[1] + bbox_placed_render[3])),
                (255, 0, 0),  # Blue color
                2  # Line thickness
            )
        
        # Draw current movable bbox (red)
        if self.current_bbox and len(self.current_bbox) == 4: 
            cv2.rectangle(
                image_to_render,
                (int(self.current_bbox[0]), int(self.current_bbox[1])),
                (int(self.current_bbox[0] + self.current_bbox[2]), 
                 int(self.current_bbox[1] + self.current_bbox[3])),
                (0, 0, 255),  # Red color
                2  # Line thickness
            )
                     
        if mode == 'rgb_array':
            return image_to_render
        elif mode == 'human':
            cv2.imshow('ROI Detection Environment', image_to_render)
            cv2.waitKey(1)
            return image_to_render

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get the current observation with history features."""
        # Clean image - WITHOUT drawn boxes!
        image = self.current_image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Placed boxes as structured data
        bbox_state = np.zeros((100, 4), dtype=np.float32)
        for i, bbox in enumerate(self.bboxes):
            if i < 100:
                bbox_state[i] = [
                    bbox[0] / self.image_size[0],
                    bbox[1] / self.image_size[1], 
                    bbox[2] / self.image_size[0],
                    bbox[3] / self.image_size[1]
                ]
        
        # Current box as structured data
        current_bbox_norm = np.array([
            self.current_bbox[0] / self.image_size[0],
            self.current_bbox[1] / self.image_size[1],
            self.current_bbox[2] / self.image_size[0], 
            self.current_bbox[3] / self.image_size[1]
        ], dtype=np.float32)
        
        # NEW: Action history (convert -1 to 7 for valid range)
        action_hist = np.array(list(self.action_history), dtype=np.int32)
        action_hist = np.where(action_hist == -1, 7, action_hist)
        
        # NEW: Position history
        if self.position_history is not None:
            pos_hist = np.array(list(self.position_history), dtype=np.float32)
        else:
            pos_hist = np.zeros((1,), dtype=np.float32)
        
        # NEW: Movement pattern features
        movement_features = self._calculate_movement_features()
        
        return {
            'image': image.astype(np.uint8),
            'bbox_state': bbox_state,
            'current_bbox': current_bbox_norm,
            'action_history': action_hist,
            'position_history': pos_hist,
            'movement_features': movement_features
        }

    def _calculate_movement_features(self) -> np.ndarray:
        """
        NEW: Calculate movement pattern features for the agent.
        
        Returns:
            np.ndarray: Array of 6 movement features
        """
        features = np.zeros(6, dtype=np.float32)
        
        if len(self.action_history) < 2:
            return features
        
        recent_actions = [a for a in list(self.action_history)[-5:] if a != -1]
        
        if len(recent_actions) < 2:
            return features
        
        # Feature 0: Movement action ratio (0-1)
        movement_actions = sum(1 for a in recent_actions if a < 4)
        features[0] = movement_actions / len(recent_actions) if recent_actions else 0
        
        # Feature 1: Action repetition (0-1, higher = more repetitive)
        if len(recent_actions) > 1:
            repeats = sum(1 for i in range(1, len(recent_actions)) 
                         if recent_actions[i] == recent_actions[i-1])
            features[1] = repeats / (len(recent_actions) - 1)
        
        # Feature 2-3: Direction consistency (-1 to 1)
        directions = [a for a in recent_actions if a < 4]
        if len(directions) > 1:
            # Check if mostly moving in consistent directions
            up_down = sum(1 if d in [0, 1] else -1 for d in directions)
            left_right = sum(1 if d in [2, 3] else -1 for d in directions)
            features[2] = abs(up_down) / len(directions)  # Vertical consistency
            features[3] = abs(left_right) / len(directions)  # Horizontal consistency
        
        # Feature 4: Recent place/remove activity
        place_remove_actions = sum(1 for a in recent_actions if a in [4, 5])
        features[4] = place_remove_actions / len(recent_actions) if recent_actions else 0
        
        # Feature 5: Oscillation detection (back-and-forth movement)
        if len(recent_actions) >= 4:
            oscillations = 0
            for i in range(len(recent_actions) - 3):
                if (recent_actions[i] < 4 and recent_actions[i+2] < 4 and
                    ((recent_actions[i] == 0 and recent_actions[i+2] == 1) or
                     (recent_actions[i] == 1 and recent_actions[i+2] == 0) or
                     (recent_actions[i] == 2 and recent_actions[i+2] == 3) or
                     (recent_actions[i] == 3 and recent_actions[i+2] == 2))):
                    oscillations += 1
            features[5] = oscillations / max(1, len(recent_actions) - 3)
        
        return features

    # ORIGINAL METHODS - NO CHANGES TO REWARD LOGIC

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

    def _move_bbox(self, direction: int) -> None:
        """
        Move the current bounding box in the specified direction.
        
        Args:
            direction: Direction to move (0: Up, 1: Down, 2: Left, 3: Right)
        """
        # Define step size for movement
        step_size = 8
        
        # Validate bbox_size and current_bbox before moving
        if (self.bbox_size is None or len(self.bbox_size) != 2 or 
            self.current_bbox is None or len(self.current_bbox) != 4):
            print("Warning: bbox_size or current_bbox is invalid in _move_bbox.")
            return
        
        # Move in the specified direction, clamping to image boundaries
        if direction == 0:  # Up
            self.current_bbox[1] = max(0, self.current_bbox[1] - step_size)
                
        elif direction == 1:  # Down
            bottom_edge = self.image_size[1] - self.bbox_size[1]
            self.current_bbox[1] = min(bottom_edge, self.current_bbox[1] + step_size)
                
        elif direction == 2:  # Left
            self.current_bbox[0] = max(0, self.current_bbox[0] - step_size)
                
        elif direction == 3:  # Right
            right_edge = self.image_size[0] - self.bbox_size[0]
            self.current_bbox[0] = min(right_edge, self.current_bbox[0] + step_size)

    def _place_bbox(self) -> float:
        """
        Place the current bounding box and calculate the resulting reward.
        
        Returns:
            float: The reward for this action
        """
        # Check if already at max boxes (99)
        if len(self.bboxes) >= 99:
            return -10.0  # Penalty for exceeding limit
        
        # Create a copy of the current bbox to place
        current_bbox_to_place = self.current_bbox.copy()
        
        # Place the bbox (add to list)
        self.bboxes.append(current_bbox_to_place)
        
        # Find which annotations this bbox covers (IoU > threshold)
        newly_covered_annotations = []
        annotations = self.current_sample.get('annotations', [])
        
        for i, ann in enumerate(annotations):
            # Skip if already covered by another bbox
            if i in self.covered_annotations:
                continue
                
            # Check IoU with current bbox
            iou = self._calculate_iou(current_bbox_to_place, ann['bbox'])
            if iou > self.iou_threshold:
                newly_covered_annotations.append(i)
                self.covered_annotations.add(i)
        
        # Calculate reward
        if len(newly_covered_annotations) == 0:
            return -1.0  # Penalty for placing empty bbox
        else:
            return float(len(newly_covered_annotations))  # +1 for each newly covered annotation

    def _remove_bbox(self) -> float:
        """
        Remove the last placed bounding box and calculate the resulting reward.
        
        Returns:
            float: The reward for this action
        """
        # Check if there are any boxes to remove
        if not self.bboxes:
            return -1.0  # Penalty if nothing to remove
            
        # Get the bbox that will be removed (last one)
        bbox_to_remove = self.bboxes[-1]
        
        # Find which annotations this bbox was covering
        annotations_in_removed_bbox = []
        annotations = self.current_sample.get('annotations', [])
        
        for i, ann in enumerate(annotations):
            iou = self._calculate_iou(bbox_to_remove, ann['bbox'])
            if iou > self.iou_threshold:
                annotations_in_removed_bbox.append(i)
        
        # Remove the box
        self.bboxes.pop()
        
        # Update covered_annotations: remove annotations that are no longer covered by any bbox
        for ann_idx in annotations_in_removed_bbox:
            # Check if this annotation is still covered by another bbox
            still_covered = False
            for bbox in self.bboxes:
                iou = self._calculate_iou(bbox, annotations[ann_idx]['bbox'])
                if iou > self.iou_threshold:
                    still_covered = True
                    break
            
            # If not covered by any other bbox, remove from covered set
            if not still_covered:
                self.covered_annotations.discard(ann_idx)
        
        # Calculate reward
        if len(annotations_in_removed_bbox) == 0:
            return 1.0  # Reward for removing empty bbox
        else:
            return -float(len(annotations_in_removed_bbox))  # -1 for each annotation that was in the bbox

    def _calculate_final_reward(self) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate final reward based on efficiency and coverage.
        
        Returns:
            Tuple containing:
                - float: The final reward value
                - Dict: Metrics about the episode performance
        """
        annotations = self.current_sample.get('annotations', [])
        total_annotations = len(annotations)
        covered_annotations_count = len(self.covered_annotations)
        num_boxes_placed = len(self.bboxes)
        
        # Calculate metrics
        if num_boxes_placed == 0:
            final_reward_val = - total_annotations  # Penalty for no boxes placed
            efficiency_ratio = 0.0
            empty_boxes = 0
            boxes_with_annotations = 0
        else:
            # Count boxes with annotations
            boxes_with_annotations = 0
            for bbox in self.bboxes:
                has_annotation = False
                for ann in annotations:
                    if self._calculate_iou(bbox, ann['bbox']) > self.iou_threshold:
                        has_annotation = True
                        break
                if has_annotation:
                    boxes_with_annotations += 1
            
            empty_boxes = num_boxes_placed - boxes_with_annotations
            
            # Calculate efficiency ratio
            efficiency_ratio = covered_annotations_count / num_boxes_placed
            
            # Calculate penalties
            empty_box_penalty = empty_boxes
            missed_annotations = total_annotations - covered_annotations_count
            missed_annotation_penalty = missed_annotations
            
            # Final reward formula
            final_reward_val = (efficiency_ratio * 10) - empty_box_penalty - missed_annotation_penalty
        
        # Coverage score for metrics
        coverage_score = covered_annotations_count / total_annotations if total_annotations > 0 else 1.0
        
        # Collect metrics for logging
        calculated_metrics = {
            'coverage_score': coverage_score,
            'efficiency_ratio': efficiency_ratio,
            'covered_annotations': covered_annotations_count,
            'total_annotations': total_annotations,
            'placed_count': num_boxes_placed,
            'empty_boxes': empty_boxes,
            'boxes_with_annotations': boxes_with_annotations,
            'missed_annotations': total_annotations - covered_annotations_count
        }
        
        return final_reward_val, calculated_metrics

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

    def action_masks(self) -> np.ndarray:
        """
        Return a mask of valid actions for the current state.
        All actions are always valid in this environment.
        
        Returns:
            np.ndarray: Binary mask where 1 indicates a valid action
        """
        return np.ones(self.action_space.n, dtype=np.int8)

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