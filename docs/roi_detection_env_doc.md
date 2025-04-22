# ROIDetectionEnv.py - Detailed Documentation

## Overview
`ROIDetectionEnv.py` implements a custom Gym environment for the ROI detection task. The environment allows an agent to learn how to optimally place bounding boxes to cover annotated objects in images, using a K-means clustering approach to determine optimal ROI placements.

## Class: ROIDetectionEnv

### Initialization
```python
def __init__(self, dataset, crop_size=(640,640), time_limit=120):
```

- **Lines 5-26**: Initialize the environment
  - **Line 12**: Call parent Gym environment initializer
  - **Lines 14-19**: Store parameters
    - `dataset`: ROIDataset instance for loading images and annotations
    - `crop_size`: Size of the ROIs to place (width, height)
    - `time_limit`: Time limit for an episode in seconds
  - **Lines 21-26**: Initialize environment state variables
    - `current_sample`: Will hold the current sample from the dataset
    - `bbox_size`: Will hold the visual bbox size for the resized image
    - `bboxes`: Empty list to store placed bounding boxes
    - `start_time`: Timer for episode time limit

- **Lines 28-36**: Define the action space
  - 0-3: Move bbox (up, down, left, right)
  - 4: Place bbox
  - 5: Remove bbox
  - 6: End episode

- **Lines 38-42**: Define the observation space
  - `image`: Box space for the RGB image (640x640x3)
  - `bbox_state`: Box space for state of placed bounding boxes (max 100 boxes)

- **Line 45**: Initialize `optimal_rois` to None (will be calculated during reset)

### Environment Reset
```python
def reset(self):
```

- **Lines 47-74**: Reset the environment with a new image
  - **Lines 48-49**: Start the timer
  - **Lines 51-56**: Get a new sample from the dataset
  - **Line 58**: Store the current image
  - **Line 61**: Calculate the visual bbox size for the resized image
  - **Line 64**: Calculate optimal ROIs using K-means
  - **Line 67**: Reset the list of placed bounding boxes
  - **Lines 70-74**: Set the initial bbox position to center of image
  - **Line 76**: Return the initial observation

### Environment Step
```python
def step(self, action):
```

- **Lines 78-109**: Process an action and return the new state
  - **Line 80-82**: Initialize return values
  - **Lines 85-93**: Check if time limit exceeded
    - End the episode if time is up
    - Calculate final reward and metrics
    - Return observation, reward, done flag, and info
  - **Lines 95-103**: Process the action
    - 0-3: Move the current bounding box
    - 4: Place the current bounding box
    - 5: Remove the last placed bounding box
    - 6: End the episode
  - **Line 108**: Return the new observation, reward, done flag, and info

### Observation Generation
```python
def _get_observation(self):
```

- **Lines 111-143**: Generate the current observation
  - **Line 112**: Copy the current image
  - **Lines 115-119**: Draw all placed bounding boxes in green
  - **Lines 122-126**: Draw the current movable bounding box in blue
  - **Lines 128-134**: Prepare the bounding box state tensor
    - Convert box coordinates to normalized values
    - Store up to 100 boxes
  - **Lines 136-142**: Return the observation as a dictionary

### Bounding Box Manipulation
```python
def _move_bbox(self, direction):
def _place_bbox(self):
def _remove_bbox(self):
```

- **Lines 145-157**: Move the current bounding box
  - Move 8 pixels in the specified direction
  - Constrain movement to stay within image bounds
- **Lines 159-166**: Place the current bounding box
  - Check for excessive overlap with existing boxes
  - Apply penalty for overlap (return -1)
  - Add the box to the list of placed boxes
  - Return neutral reward (0) for successful placement
- **Lines 168-174**: Remove the last placed bounding box
  - Return penalty (-0.1) if no boxes exist
  - Remove the last box
  - Return neutral reward (0) for successful removal

### Utility Functions
```python
def _calculate_iou(self, box1, box2):
def _scale_bbox_to_original(self, bbox):
def _is_bbox_contained(self, bbox1, bbox2, threshold=0.8):
```

- **Lines 176-203**: Calculate Intersection over Union (IoU) between two boxes
  - Convert from [x, y, w, h] format to [x1, y1, x2, y2]
  - Calculate intersection area
  - Calculate union area
  - Return the IoU ratio
- **Lines 205-213**: Scale a bbox from resized image to original image dimensions
  - Apply inverse of the scale factors from dataset
- **Lines 215-237**: Check if one bbox is mostly contained within another
  - Calculate intersection area
  - Check if intersection covers at least threshold (80%) of first bbox

### Optimal ROI Calculation
```python
def _calculate_optimal_rois(self):
```

- **Lines 239-322**: Implement K-means based ROI discovery
  - **Lines 240-245**: Extract relevant information
    - Annotations from current sample
    - Image dimensions
    - ROI dimensions and half dimensions
  - **Lines 248-249**: Define helper to clamp values
  - **Lines 252-256**: Collect annotation centers
    - For each annotation, compute the center point (x+w/2, y+h/2)
  - **Lines 259-266**: Handle special cases
    - For 0 or 1 annotations, return a single centered ROI
  - **Line 269**: Convert points to numpy array for K-means
  - **Lines 272-307**: Search for minimal k (number of clusters)
    - Try k from 1 to number of points
    - For each k, fit K-means to annotation centers
    - Check if all points within each cluster can be covered by a single ROI
    - Stop when a valid k is found
  - **Lines 310-320**: Build ROIs around cluster centers
    - Center ROIs on cluster centers
    - Clamp ROI centers to ensure they remain within the image
    - Convert to [x, y, width, height] format

### Evaluation Functions
```python
def _calculate_annotation_coverage(self, rois, annotations):
def _calculate_roi_matching(self, placed_rois, optimal_rois):
def _calculate_roi_overlap_penalty(self, rois):
```

- **Lines 324-341**: Calculate annotation coverage
  - Count how many annotations are fully covered by at least one ROI
  - Return the proportion of covered annotations
- **Lines 343-360**: Calculate ROI matching score
  - For each optimal ROI, find best matching placed ROI using IoU
  - Average the IoU scores across all optimal ROIs
- **Lines 362-376**: Calculate penalty for overlapping ROIs
  - For each pair of ROIs, calculate IoU
  - Apply penalty for IoU values above 0.3
  - Cap total penalty at 1.0

### Final Reward Calculation
```python
def _calculate_final_reward(self):
```

- **Lines 378-430**: Calculate the final reward
  - **Lines 379-381**: Apply penalty if no ROIs were placed
  - **Lines 384-385**: Calculate optimal ROIs if not already done
  - **Lines 388-389**: Calculate coverage of annotations
  - **Line 392**: Calculate how well placed ROIs match optimal ones
  - **Lines 395-398**: Calculate efficiency score based on ROI count
  - **Line 401**: Calculate overlap penalty for placed ROIs
  - **Lines 404-408**: Define weights for different components
  - **Lines 410-415**: Compute final weighted reward
  - **Lines 418-429**: Prepare metrics dictionary for debugging
  - **Line 431**: Return final reward and metrics

### Visualization Functions
```python
def _get_bbox_size(self):
def render(self, mode='rgb_array'):
```

- **Lines 433-438**: Calculate the bbox size for the resized image
  - Apply scale factors to the crop size
- **Lines 440-485**: Render the current state of the environment
  - Draw ground truth annotations in yellow
  - Draw optimal ROIs in red
  - Draw placed bounding boxes in green
  - Draw current movable bounding box in blue
  - Return image for different rendering modes

## Key Features

1. **Reinforcement Learning Integration**:
   - Follows the Gym environment interface
   - Provides clear state, action, and reward definitions
   - Handles episode termination conditions

2. **K-means Based Optimization**:
   - Determines optimal ROI placement using clustering
   - Minimizes the number of ROIs while covering all annotations
   - Constrains clusters to ensure annotations can be covered

3. **Comprehensive Evaluation Metrics**:
   - Coverage score: Percentage of annotations covered
   - ROI matching score: Similarity to optimal placement
   - Efficiency score: Optimal use of ROIs
   - Overlap penalty: Discourages redundant ROIs

4. **Visualization Support**:
   - Renders environment state with clear visual cues
   - Distinguishes between different types of boxes using color

## Methodology

The environment implements a comprehensive methodology for ROI placement:

1. **Optimal ROI Discovery**: Uses K-means clustering to find the minimal number of ROIs that can cover all annotations, subject to the constraint that all points within a cluster must fit within a single ROI.

2. **Agent Training Environment**: Provides a structured environment where an agent can learn to place ROIs effectively through exploration and reward signals.

3. **Multi-faceted Reward System**: Evaluates ROI placement based on multiple criteria:
   - Annotation coverage (primary objective)
   - Match with optimal placement (strategic guidance)
   - Efficiency in ROI count (resource optimization)
   - Avoidance of excessive overlap (redundancy reduction)

4. **Time-constrained Operation**: Implements a realistic time limit for ROI placement to encourage efficient decision-making.

This approach enables the learning of ROI placement strategies that balance coverage, efficiency, and optimality in a principled manner.
