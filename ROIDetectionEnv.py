import numpy as np
import gym
from gym import spaces
import cv2
from sklearn.cluster import KMeans

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
        
        # Observation space: image and current bbox state
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(self.image_size[1], self.image_size[0], 3), dtype=np.uint8),
            'bbox_state': spaces.Box(low=0, high=1, shape=(100, 4), dtype=np.float32)  # max 100 bboxes
        })

        # Store optimal ROIs
        self.optimal_rois = None

    def reset(self):
        """Reset the environment with a new image"""
        import time
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
        
        # Calculate optimal ROIs using K-means
        self.optimal_rois = self._calculate_optimal_rois()

        # Reset bboxes
        self.bboxes = []

        # Initial bbox center
        self.current_bbox = [
            (self.image_size[0] - self.bbox_size[0]) // 2,
            (self.image_size[1] - self.bbox_size[1]) // 2,
            self.bbox_size[0], self.bbox_size[1]
        ]

        return self._get_observation()

    def step(self, action):
        """Take a step in the environment based on the action"""
        import time
        
        reward = 0
        done = False
        info = {}
        
        # Check if time limit exceeded
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.time_limit:
            done = True
            reward, metrics = self._calculate_final_reward()
            metrics['time'] = {
                'elapsed': elapsed_time,
                'limit': self.time_limit
            }
            info['metrics'] = metrics
            return self._get_observation(), reward, done, info
        
        if action < 4:  # Move bbox
            self._move_bbox(action)
        elif action == 4:  # Place bbox
            reward = self._place_bbox()
        elif action == 5:  # Remove bbox
            reward = self._remove_bbox()
        else:  # End episode
            done = True
            reward, metrics = self._calculate_final_reward()
            metrics['time'] = {
                'elapsed': elapsed_time,
                'limit': self.time_limit
            }
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

    def _calculate_optimal_rois(self):
        """
        KMeans-based ROI discovery under L∞ constraint.
        
        Returns: list of [x0, y0, roi_w, roi_h]
        """
        annotations = self.current_sample['annotations']
        H, W = self.current_image.shape[:2]
        roi_w, roi_h = self.bbox_size
        half_w, half_h = roi_w/2.0, roi_h/2.0
        
        # Helper to clamp center so ROI stays in image
        def clamp(val, lo, hi):
            return max(lo, min(val, hi))
            
        # 1) collect annotation centers
        pts = []
        for ann in annotations:
            x, y, w, h = ann['bbox']
            pts.append((x + w/2.0, y + h/2.0))
        
        # If no annotations or just one, return a single centered ROI
        if len(pts) <= 1:
            if len(pts) == 1:
                cx, cy = pts[0]
                cx = clamp(cx, half_w, W - half_w)
                cy = clamp(cy, half_h, H - half_h)
            else:
                cx, cy = W/2, H/2
            return [[cx - half_w, cy - half_h, roi_w, roi_h]]
            
        # Convert to numpy array for KMeans
        pts = np.array(pts)
        
        # 2) search for minimal k
        best_k = len(pts)
        best_centers = None
        
        for k in range(1, len(pts)+1):
            # Skip if we have too few points for k clusters
            if len(pts) < k:
                continue
                
            # Fit KMeans
            km = KMeans(n_clusters=k, random_state=0).fit(pts)
            centers = km.cluster_centers_
            labels = km.labels_
            
            good = True
            # for each cluster, check L∞ radius
            for ci in range(k):
                cx, cy = centers[ci]
                # all pts in this cluster
                members = [pts[i] for i, lab in enumerate(labels) if lab == ci]
                if not members:  # Skip empty clusters
                    continue
                # max |Δx|, |Δy|
                max_dx = max(abs(px - cx) for px, py in members)
                max_dy = max(abs(py - cy) for px, py in members)
                if max_dx > half_w or max_dy > half_h:
                    good = False
                    break
            
            if good:
                best_k = k
                best_centers = centers
                break
        
        # 3) build ROIs around those cluster-centers
        rois = []
        for (cx, cy) in best_centers:
            # clamp inside image
            cx = clamp(cx, half_w, W - half_w)
            cy = clamp(cy, half_h, H - half_h)
            x0, y0 = cx - half_w, cy - half_h
            rois.append([x0, y0, roi_w, roi_h])
        
        return rois

    def _is_bbox_contained(self, bbox1, bbox2, threshold=0.8):
        """Check if bbox1 is mostly contained within bbox2"""
        # Convert to [x1, y1, x2, y2] format
        bbox1_x1, bbox1_y1 = bbox1[0], bbox1[1]
        bbox1_x2, bbox1_y2 = bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]
        
        bbox2_x1, bbox2_y1 = bbox2[0], bbox2[1]
        bbox2_x2, bbox2_y2 = bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]
        
        # Calculate intersection area
        x_left = max(bbox1_x1, bbox2_x1)
        y_top = max(bbox1_y1, bbox2_y1)
        x_right = min(bbox1_x2, bbox2_x2)
        y_bottom = min(bbox1_y2, bbox2_y2)
        
        if x_right < x_left or y_bottom < y_top:
            return False
                
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bbox1_area = (bbox1_x2 - bbox1_x1) * (bbox1_y2 - bbox1_y1)
        
        # Return true if intersection covers at least threshold of bbox1
        return intersection_area / bbox1_area >= threshold

    def _calculate_annotation_coverage(self, rois, annotations):
        """Calculate how many annotations are properly covered by at least one ROI"""
        covered_count = 0
        total_annotations = len(annotations)
        
        if total_annotations == 0:
            return 1.0  # Perfect coverage if no annotations
        
        for ann in annotations:
            ann_bbox = ann['bbox']  # [x, y, w, h]
            # Check if this annotation is covered by any ROI
            for roi in rois:
                if self._is_bbox_contained(ann_bbox, roi):
                    covered_count += 1
                    break
        
        return covered_count / total_annotations

    def _calculate_roi_matching(self, placed_rois, optimal_rois):
        """Calculate how well placed ROIs match optimal ROIs using IoU"""
        if not optimal_rois or not placed_rois:
            return 0.0
        
        # For each optimal ROI, find best matching placed ROI
        total_iou = 0.0
        
        for opt_roi in optimal_rois:
            best_iou = 0.0
            for placed_roi in placed_rois:
                iou = self._calculate_iou(opt_roi, placed_roi)
                best_iou = max(best_iou, iou)
            total_iou += best_iou
        
        # Average IoU across all optimal ROIs
        return total_iou / len(optimal_rois)

    def _calculate_roi_overlap_penalty(self, rois):
        """Calculate penalty for excessive overlap between ROIs"""
        if len(rois) <= 1:
            return 0.0
        
        total_penalty = 0.0
        for i in range(len(rois)):
            for j in range(i+1, len(rois)):
                iou = self._calculate_iou(rois[i], rois[j])
                # Only penalize IoU above 0.3
                if iou > 0.3:
                    total_penalty += (iou - 0.3)
        
        return min(1.0, total_penalty)  # Cap penalty at 1.0

    def _calculate_final_reward(self):
        """Calculate final reward based on optimal ROIs and annotation coverage"""
        if not self.bboxes:
            return -10.0, {"metrics": "No ROIs placed"}  # Large penalty for not placing any ROIs
        
        # Recalculate optimal ROIs (in case they weren't calculated during reset)
        if self.optimal_rois is None:
            self.optimal_rois = self._calculate_optimal_rois()
        
        # Calculate coverage of annotations
        annotations = self.current_sample['annotations']
        coverage_score = self._calculate_annotation_coverage(self.bboxes, annotations)
        
        # Calculate how well our ROIs match optimal ones
        roi_matching_score = self._calculate_roi_matching(self.bboxes, self.optimal_rois)
        
        # Efficiency: reward for using close to optimal number of ROIs
        optimal_count = len(self.optimal_rois)
        placed_count = len(self.bboxes)
        efficiency_score = max(0, 1 - (abs(placed_count - optimal_count) / max(1, optimal_count)))
        
        # Calculate overlap among placed ROIs (penalize excessive overlap)
        overlap_penalty = self._calculate_roi_overlap_penalty(self.bboxes)
        
        # Weights for different components
        coverage_weight = 5.0     # Highest priority: cover all annotations
        matching_weight = 3.0     # Important: match optimal placement
        efficiency_weight = 2.0   # Somewhat important: use right number of ROIs
        overlap_penalty_weight = 1.5  # Penalize excessive overlap
        
        final_reward = (
            coverage_score * coverage_weight +
            roi_matching_score * matching_weight +
            efficiency_score * efficiency_weight -
            overlap_penalty * overlap_penalty_weight
        )
        
        # Include metrics for debugging/visualization
        metrics = {
            'optimal_rois': self.optimal_rois,
            'coverage_score': coverage_score,
            'roi_matching_score': roi_matching_score,
            'efficiency_score': efficiency_score,
            'overlap_penalty': overlap_penalty,
            'optimal_count': optimal_count,
            'placed_count': placed_count
        }
        
        return final_reward, metrics
        
    def _get_bbox_size(self):
        """
        Calculate the bbox size (width, height) in the resized image
        that corresponds to crop_size in the original image.
        """
        scale_w, scale_h = self.current_sample['scale_factors']
        crop_w, crop_h = self.crop_size
        return int(crop_w * scale_w), int(crop_h * scale_h)

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
        
        # Draw optimal ROIs in red
        if self.optimal_rois:
            for roi in self.optimal_rois:
                cv2.rectangle(image,
                             (int(roi[0]), int(roi[1])),
                             (int(roi[0] + roi[2]), int(roi[1] + roi[3])),
                             (0, 0, 255), 1)  # Red color
                
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