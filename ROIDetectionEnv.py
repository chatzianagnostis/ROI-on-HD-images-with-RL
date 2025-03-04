import numpy as np
import gym
from gym import spaces
import cv2
from ultralytics import YOLO

class ROIDetectionEnv(gym.Env):
    def __init__(self, dataset, bbox_size=(32, 32), yolo_model_path='yolov8n.pt'):
        """
        Initialize the ROI Detection Environment.
        
        Args:
            dataset: ROIDataset instance for loading images and annotations
            bbox_size: Size of the bounding boxes to place (width, height)
            yolo_model_path: Path to the YOLO model
        """
        super(ROIDetectionEnv, self).__init__()
        
        self.dataset = dataset
        self.image_size = dataset.image_size
        self.bbox_size = bbox_size
        self.current_sample = None
        self.bboxes = []
        
        # Define action space:
        # 0-3: Move bbox (up, down, left, right)
        # 4: Place bbox
        # 5: Remove bbox
        # 6: End episode
        self.action_space = spaces.Discrete(7)

        # Load YOLO
        self.detector = YOLO(yolo_model_path)
        
        # Observation space: image and current bbox state
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(self.image_size[1], self.image_size[0], 3), dtype=np.uint8),
            'bbox_state': spaces.Box(low=0, high=1, shape=(100, 4), dtype=np.float32)  # max 100 bboxes
        })

        # Store full image detection results
        self.full_image_results = None

    def reset(self):
        """Reset the environment with a new image"""
        # Get next sample from dataset
        self.current_sample = next(self.dataset)
        self.current_image = self.current_sample['resized_image']
        
        # Run YOLO on the full resized image to get baseline accuracy
        self.full_image_results = self.detector(self.current_image)
        
        # Reset bounding boxes
        self.bboxes = []
        
        # Initial bbox position: center of the image
        self.current_bbox = [
            (self.image_size[0] - self.bbox_size[0]) // 2,
            (self.image_size[1] - self.bbox_size[1]) // 2,
            self.bbox_size[0], self.bbox_size[1]
        ]
        
        return self._get_observation()

    def step(self, action):
        """Take a step in the environment based on the action"""
        reward = 0
        done = False
        info = {}
        
        if action < 4:  # Move bbox
            self._move_bbox(action)
        elif action == 4:  # Place bbox
            reward = self._place_bbox()
        elif action == 5:  # Remove bbox
            reward = self._remove_bbox()
        else:  # End episode
            done = True
            reward, metrics = self._calculate_final_reward()
            info['metrics'] = metrics
        
        return self._get_observation(), reward, done, info

    def _get_observation(self):
        """Get the current observation"""
        image = self.current_image.copy()
        
        # Draw all bboxes
        for bbox in self.bboxes:
            cv2.rectangle(image, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                         (0, 255, 0), 2)
        
        # Draw current bbox
        cv2.rectangle(image,
                     (int(self.current_bbox[0]), int(self.current_bbox[1])),
                     (int(self.current_bbox[0] + self.current_bbox[2]), 
                      int(self.current_bbox[1] + self.current_bbox[3])),
                     (255, 0, 0), 2)
        
        bbox_state = np.zeros((100, 4))
        for i, bbox in enumerate(self.bboxes):
            if i < 100:
                bbox_state[i] = [bbox[0]/self.image_size[0], 
                               bbox[1]/self.image_size[1],
                               bbox[2]/self.image_size[0],
                               bbox[3]/self.image_size[1]]
        
        return {
            'image': image,
            'bbox_state': bbox_state
        }

    def _move_bbox(self, direction):
        """Move the current bounding box"""
        step_size = 8
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
        """Place the current bounding box"""
        # Check for overlap with existing bboxes
        for bbox in self.bboxes:
            if self._calculate_iou(self.current_bbox, bbox) > 0.3:
                return -1  # Penalty for overlap
        
        self.bboxes.append(self.current_bbox.copy())
        return 0  # Neutral reward for placement

    def _remove_bbox(self):
        """Remove the last placed bounding box"""
        if not self.bboxes:
            return -0.1  # Penalty for trying to remove when no bbox exists
        
        self.bboxes.pop()
        return 0  # Neutral reward for removal

    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes"""
        # Convert to [x1, y1, x2, y2] format
        box1_x1, box1_y1 = box1[0], box1[1]
        box1_x2, box1_y2 = box1[0] + box1[2], box1[1] + box1[3]
        
        box2_x1, box2_y1 = box2[0], box2[1]
        box2_x2, box2_y2 = box2[0] + box2[2], box2[1] + box2[3]
        
        # Calculate intersection area
        x_left = max(box1_x1, box2_x1)
        y_top = max(box1_y1, box2_y1)
        x_right = min(box1_x2, box2_x2)
        y_bottom = min(box1_y2, box2_y2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        union_area = box1_area + box2_area - intersection_area
        
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

    def _evaluate_roi(self):
        """Evaluate the ROIs by comparing YOLO detection on ROIs vs full image"""
        if not self.bboxes:
            return {'full': {}, 'roi': {}}, 0.0  # No ROIs to evaluate
            
        # Get detection results from the full image
        full_image_metrics = self._get_detection_metrics(self.full_image_results)
        
        # Process each ROI
        roi_images = []
        original_rois = []
        
        original_image = self.current_sample['original_image']
        
        for bbox in self.bboxes:
            # Scale bbox to original image dimensions
            original_bbox = self._scale_bbox_to_original(bbox)
            original_rois.append(original_bbox)
            
            # Extract ROI from original image
            x, y, w, h = original_bbox
            # Ensure we don't go out of bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, original_image.shape[1] - x)
            h = min(h, original_image.shape[0] - y)
            
            roi = original_image[y:y+h, x:x+w]
            if roi.size > 0:  # Only append valid ROIs
                roi_images.append(roi)
        
        if not roi_images:
            return {'full': full_image_metrics, 'roi': {}}, -10.0  # Invalid ROIs
            
        # Run YOLO on all ROIs
        roi_results = [self.detector(roi) for roi in roi_images]
        
        # Calculate average metrics across all ROIs
        roi_metrics_list = [self._get_detection_metrics(result) for result in roi_results]
        avg_roi_metrics = {
            'precision': np.mean([m.get('precision', 0.0) for m in roi_metrics_list]),
            'recall': np.mean([m.get('recall', 0.0) for m in roi_metrics_list]),
            'map50': np.mean([m.get('map50', 0.0) for m in roi_metrics_list]),
        }
        
        # Combine metrics into a dictionary to return
        metrics = {
            'full': full_image_metrics,
            'roi': avg_roi_metrics,
            'roi_count': len(self.bboxes),
            'coverage': sum(bbox[2] * bbox[3] for bbox in self.bboxes) / (self.image_size[0] * self.image_size[1])
        }
        
        return metrics, avg_roi_metrics

    def _get_detection_metrics(self, results):
        """Extract precision, recall, mAP50 from YOLO results"""
        # Note: This is a simplified version, as actual metrics calculation
        # depends on the specific YOLO implementation
        if not results or len(results) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'map50': 0.0}
            
        metrics = {}
        
        try:
            # Try to extract metrics from the results
            # Different YOLO versions might have different ways to access metrics
            if hasattr(results[0], 'box') and hasattr(results[0].box, 'metrics'):
                box_metrics = results[0].box.metrics
                metrics = {
                    'precision': box_metrics.get('precision', 0.0),
                    'recall': box_metrics.get('recall', 0.0),
                    'map50': box_metrics.get('map50', 0.0),
                }
            else:
                # Alternative way to estimate metrics based on detections
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    # Simplified metrics based on detection count and confidence
                    confidence_scores = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else []
                    metrics = {
                        'precision': np.mean(confidence_scores) if len(confidence_scores) > 0 else 0.0,
                        'recall': min(1.0, len(boxes) / 10),  # Simplified recall estimate
                        'map50': np.mean(confidence_scores) if len(confidence_scores) > 0 else 0.0,
                    }
        except Exception as e:
            print(f"Error extracting metrics from YOLO results: {e}")
            metrics = {'precision': 0.0, 'recall': 0.0, 'map50': 0.0}
            
        return metrics

    def _calculate_final_reward(self):
        """Calculate final reward based on the difference between ROI accuracy and full image accuracy"""
        metrics, roi_metrics = self._evaluate_roi()
        
        if not self.bboxes:
            return -10.0, metrics  # Large penalty for not placing any ROIs
        
        # Get the metrics from the results
        full_metrics = metrics['full']
        
        # Calculate the difference in metrics
        precision_diff = roi_metrics.get('precision', 0.0) - full_metrics.get('precision', 0.0)
        recall_diff = roi_metrics.get('recall', 0.0) - full_metrics.get('recall', 0.0)
        map_diff = roi_metrics.get('map50', 0.0) - full_metrics.get('map50', 0.0)
        
        # Coverage reward: proportion of the image covered by ROIs
        total_roi_area = sum(bbox[2] * bbox[3] for bbox in self.bboxes)
        image_area = self.image_size[0] * self.image_size[1]
        coverage_ratio = min(1.0, total_roi_area / image_area)
        
        # Efficiency reward: higher if we achieve similar or better results with less coverage
        efficiency_factor = max(0.1, (1.0 - coverage_ratio) * 2)
        
        # Combine metrics into final reward
        # Weighted combination of metric differences and efficiency
        metric_weight = 5.0
        efficiency_weight = 10.0
        
        # The core reward is based on the average improvement across metrics
        metrics_avg_diff = (precision_diff + recall_diff + map_diff) / 3.0
        
        # Penalize excessive ROIs - we want minimal ROIs that capture key information
        roi_penalty = -0.2 * max(0, len(self.bboxes) - 5)
        
        final_reward = (metrics_avg_diff * metric_weight) + (efficiency_factor * efficiency_weight) + roi_penalty
        
        return final_reward, metrics

    def render(self, mode='rgb_array'):
        """Render the current state of the environment"""
        if self.current_image is None:
            return None
            
        image = self.current_image.copy()
        
        # Draw ground truth annotations if available
        if 'annotations' in self.current_sample:
            for ann in self.current_sample['annotations']:
                bbox = ann['bbox']
                category_id = ann['category_id']
                # Draw ground truth in yellow
                cv2.rectangle(image, 
                             (int(bbox[0]), int(bbox[1])), 
                             (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                             (0, 255, 255), 1)
        
        # Draw all placed bboxes in green
        for bbox in self.bboxes:
            cv2.rectangle(image, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                         (0, 255, 0), 2)
        
        # Draw current movable bbox in blue
        cv2.rectangle(image,
                     (int(self.current_bbox[0]), int(self.current_bbox[1])),
                     (int(self.current_bbox[0] + self.current_bbox[2]), 
                      int(self.current_bbox[1] + self.current_bbox[3])),
                     (255, 0, 0), 2)
                     
        if mode == 'rgb_array':
            return image
        elif mode == 'human':
            cv2.imshow('ROI Detection Environment', image)
            cv2.waitKey(1)