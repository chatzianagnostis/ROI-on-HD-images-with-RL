import numpy as np
import gym
from gym import spaces
import cv2
from ultralytics import YOLO

class ROIDetectionEnv(gym.Env):
    def __init__(self, dataset_path, image_size=(640, 640), bbox_size=(32, 32)):
        super(ROIDetectionEnv, self).__init__()
        
        self.image_size = image_size
        self.bbox_size = bbox_size
        self.dataset_path = dataset_path
        self.current_image = None
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
            'image': spaces.Box(low=0, high=255, shape=(image_size[0], image_size[1], 3), dtype=np.uint8),
            'bbox_state': spaces.Box(low=0, high=1, shape=(100, 4), dtype=np.float32)  # max 100 bboxes
        })


    def reset(self):
        # Load new image from the dataset
        self.current_image = self._load_next_image()
        self.bboxes = []
        # Initial bbox position: center of the image
        self.current_bbox = [
            (self.image_size[0] - self.bbox_size[0]) // 2,
            (self.image_size[1] - self.bbox_size[1]) // 2,
            self.bbox_size[0], self.bbox_size[1]
        ]
        return self._get_observation()


    def step(self, action):
        reward = 0

        done = False
        
        if action < 4:  # Move bbox
            self._move_bbox(action)
        elif action == 4:  # Place bbox
            reward = self._place_bbox()
        elif action == 5:  # Remove bbox
            reward = self._remove_bbox()
        else:  # End episode
            done = True
            reward = self._calculate_final_reward()
        
        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        image = self.current_image.copy()
        
        # Draw all bboxes
        for bbox in self.bboxes:
            cv2.rectangle(image, 
                         (bbox[0], bbox[1]), 
                         (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                         (0, 255, 0), 2)
        
        # Draw current bbox
        cv2.rectangle(image,
                     (self.current_bbox[0], self.current_bbox[1]),
                     (self.current_bbox[0] + self.current_bbox[2], 
                      self.current_bbox[1] + self.current_bbox[3]),
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
        step_size = 2
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
        #  # Check for overlap with existing bboxes
        for bbox in self.bboxes:
            if self._calculate_iou(self.current_bbox, bbox) > 0.3:
                return -1  # Penalty for overlap
        
        self.bboxes.append(self.current_bbox.copy())
        return 0 # Neutral reward for placement


    def _remove_bbox(self):

        if not self.bboxes:
            return - 0.1  # Penalty for trying to remove when no bbox exists
        
        self.bboxes.pop()
        return 0 # Neutral reward for removal


    def _calculate_final_reward(self):

        # todo
        self._evaluate_roi()
        return 