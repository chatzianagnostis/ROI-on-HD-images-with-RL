# ROIDetectionEnv.py - Detailed Documentation

## Overview
`ROIDetectionEnv.py` implements a custom Gym environment for the ROI detection task. The environment allows an agent to learn how to optimally place bounding boxes to cover annotated objects in images, using a K-means clustering approach to determine optimal ROI placements.

## Class: ROIDetectionEnv

### Initialization
```python
def __init__(self, dataset, crop_size=(640,640), time_limit=120):
```

- Initialize the environment with:
  - `dataset`: ROIDataset instance for loading images and annotations
  - `crop_size`: Size of the ROIs to place (width, height)
  - `time_limit`: Time limit for an episode in seconds
- Define action space:
  - 0-3: Move bbox (up, down, left, right)
  - 4: Place bbox
  - 5: Remove bbox
  - 6: End episode
- Define observation space as a dictionary:
  - `image`: Box space for the RGB image (640x640x3)
  - `bbox_state`: Box space for state of placed bounding boxes (max 100 boxes)
- Initialize `optimal_rois` to None (calculated during reset)

### Environment Reset
```python
def reset(self):
```

- Reset the environment with a new image
- Start the episode timer
- Get a new sample from the dataset
- Calculate the visual bbox size for the resized image
- Calculate optimal ROIs using K-means
- Reset the list of placed bounding boxes
- Position the initial bbox at the center of image
- Return the initial observation

### Environment Step
```python
def step(self, action):
```

- Process an action and return the new state
- Calculate potential-based shaping reward for movement actions
- Apply reward/penalty for placing/removing boxes
- Check if time limit exceeded and end episode if necessary
- Return (observation, reward, done, info)

### Potential-Based Reward Shaping
```python
def _potential_function(self, bbox):
def _dist_to_nearest_unmatched_opt_roi(self, bbox):
```

- Implements potential-based reward shaping to guide agent toward unmatched optimal ROIs
- Calculates Manhattan distance to nearest unmatched optimal ROI
- Adds small random jitter to prevent overfitting to exact positions
- Returns higher potential value when closer to an unmatched optimal ROI

### Optimal ROI Calculation
```python
def _calculate_optimal_rois(self):
```

- Implements K-means based ROI discovery with L∞ constraint
- Collects annotation centers as clustering points
- Searches for minimal k (number of clusters) where:
  - All points in each cluster fit within a single ROI (L∞ constraint)
  - max |Δx| ≤ half_width and max |Δy| ≤ half_height
- Places ROIs centered on cluster centers, clamped to stay within image bounds
- Returns list of optimal ROIs in [x, y, width, height] format

### Evaluation Functions
```python
def _calculate_annotation_coverage(self, rois, annotations):
def _calculate_roi_matching(self, placed_rois, optimal_rois):
def _calculate_roi_overlap_penalty(self, rois):
```

- Calculate annotation coverage: percentage of annotations fully covered by ROIs
- Calculate ROI matching score: IoU-based similarity to optimal placement
- Calculate overlap penalty: penalizes excessive overlap between ROIs

### Final Reward Calculation
```python
def _calculate_final_reward(self):
```

- Calculate composite final reward with weighted components:
  - Coverage score (weight: 50.0): Percentage of annotations properly covered
  - ROI matching score (weight: 30.0): How well placed ROIs match optimal ones
  - Efficiency score (weight: 15.0): How close to optimal number of ROIs
  - Overlap penalty (weight: 5.0): Penalty for excessive ROI overlap
- Returns final reward value and detailed metrics dictionary

### Visualization
```python
def render(self, mode='rgb_array'):
def visualize_reward_landscape(self, output_path="reward_landscape.jpg"):
```

- Render environment state with color-coded boxes:
  - Yellow: Ground truth annotations
  - Red: Optimal ROIs from K-means
  - Green: Placed bounding boxes
  - Blue: Current movable bounding box
- Visualize reward landscape showing potential values across image
- Generate heatmap overlay indicating high-value regions for ROI placement

## Key Innovations

1. **K-means with L∞ Constraint**:
   - Determines optimal ROI placement using K-means clustering
   - Ensures all annotations in a cluster can be covered by a single ROI
   - Minimizes the number of ROIs while maintaining complete coverage

2. **Potential-Based Reward Shaping**:
   - Guides agent toward unmatched optimal ROIs
   - Provides smooth gradient for learning effective movement patterns
   - Adds random jitter to prevent overfitting to exact positions

3. **Multi-Component Reward System**:
   - Balances coverage, efficiency, ROI matching, and overlap
   - Weighted components prioritize complete annotation coverage
   - Detailed metrics for performance analysis

4. **Reward Landscape Visualization**:
   - Creates visual representation of the potential function
   - Helps understand agent decision-making
   - Shows high-value regions for ROI placement

## Methodology

The environment implements a comprehensive methodology for ROI placement:

1. **Optimal ROI Discovery**: Uses K-means clustering to find the minimal number of ROIs that can cover all annotations, subject to the L∞ constraint.

2. **Agent Training Environment**: Provides a structured environment where an agent can learn to place ROIs effectively through exploration and reward signals.

3. **Multi-faceted Reward System**: Evaluates ROI placement based on multiple criteria with careful weighting to prioritize coverage while encouraging efficiency.

4. **Time-constrained Operation**: Implements a realistic time limit for ROI placement to encourage efficient decision-making.

This approach enables the learning of ROI placement strategies that balance coverage, efficiency, and optimality in a principled manner.