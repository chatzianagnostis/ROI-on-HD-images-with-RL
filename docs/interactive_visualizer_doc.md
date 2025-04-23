# InteractiveBBoxPlacementVisualiser.py - Detailed Documentation

## Overview
`InteractiveBBoxPlacementVisualiser.py` implements an interactive testing tool for the K-means based ROI Detection Environment. It allows manual control or random testing of the environment, providing visual feedback and metrics on ROI placement performance compared to the optimal K-means solution.

## Class: ROIEnvTester

### Initialization
```python
def __init__(self, dataset_path, coco_json_path, crop_size=(640, 640)):
```

- Initialize the environment tester with:
  - `dataset_path`: Path to the dataset directory
  - `coco_json_path`: Path to the COCO annotations
  - `crop_size`: Size of the crop window (width, height)
- Create output directory for results
- Initialize dataset with specified parameters
- Initialize environment with K-means based ROI optimization
- Set up window name and tracking variables for actions and rewards

### Manual Episode Testing
```python
def run_manual_episode(self):
```

- Run interactive episode with user control
- Display controls and instructions:
  - W/Up: Move up
  - S/Down: Move down
  - A/Left: Move left
  - D/Right: Move right
  - Space: Place box
  - R: Remove last box
  - E: End episode
  - Q: Quit
  - K: Toggle optimal ROI visibility
  - V: Show reward landscape visualization
- Reset environment and handle user input
- Display visual feedback with:
  - Yellow: Ground truth annotations
  - Red: Optimal ROIs from K-means
  - Green: User-placed ROIs
  - Blue: Current movable ROI
- Track and display rewards
- Show detailed metrics at episode end

### Random Episode Testing
```python
def run_random_episode(self, num_steps=100, render=True):
```

- Run episode with random actions
- Use weighted probabilities for actions:
  - Movement actions: 20% each (up, down, left, right)
  - Place box: 10%
  - Remove/end: 5% each
- Display optimal ROIs from K-means alongside random actions
- Track and display metrics
- Save final state visualization

### Action Analysis
```python
def analyze_action_distribution(self):
```

- Analyze the distribution of actions and rewards
- Print action frequency statistics
- Calculate reward statistics:
  - Total, average, min, and max rewards
- Group rewards by action type
- Display average reward for each action

## Main Function and Entry Point

```python
def main():
```

- Parse command line arguments:
  - `--dataset`: Path to the dataset directory
  - `--annotations`: Path to the COCO JSON annotation file
  - `--crop_width`: Width of the crop window (default: 640)
  - `--crop_height`: Height of the crop window (default: 640)
  - `--mode`: Test mode - manual, random, or both (default: manual)
- Create tester instance with specified parameters
- Run tests based on selected mode
- Analyze action distribution after testing

## Key Innovations

1. **K-means Optimal ROI Visualization**:
   - Displays K-means generated optimal ROIs in red
   - Allows toggling visibility with K key
   - Provides visual comparison between optimal and placed ROIs

2. **Enhanced Metrics Display**:
   - Coverage score: Percentage of annotations properly covered
   - ROI matching score: How well placed ROIs match optimal ones
   - Efficiency score: How close to optimal number of ROIs
   - Overlap penalty: Penalty for excessive ROI overlap
   - Time usage metrics

3. **Reward Landscape Visualization**:
   - Added V key to visualize the reward landscape
   - Shows potential function values across the image
   - Highlights high-value regions for ROI placement

4. **Interactive Comparison**:
   - Direct visual comparison with K-means optimal solution
   - Real-time feedback on ROI placement quality
   - Detailed metrics at episode end

## Testing Modes

### 1. Manual Mode
Allows a user to directly interact with the environment using keyboard controls. This mode is useful for:
- Understanding how the environment responds to actions
- Testing specific ROI placement strategies
- Visually comparing manual placement with K-means optimal placement
- Exploring the reward landscape

### 2. Random Mode
Automatically tests the environment with random actions. This mode is useful for:
- Quickly generating baseline performance metrics
- Testing environment stability
- Collecting action-reward statistics without user bias
- Comparing random placement to K-means optimal placement

## Methodology

The visualizer tool implements a comprehensive methodology for testing and evaluating the K-means based ROI Detection Environment:

1. **Interactive Validation**: Provides a means to manually verify environment behavior and visually assess the K-means algorithm's effectiveness.

2. **Comparative Analysis**: Enables direct comparison between manual ROI placement and the automated K-means approach.

3. **Quantitative Evaluation**: Collects and analyzes metrics to objectively measure performance against the K-means baseline.

4. **Statistical Analysis**: Examines patterns in actions and rewards to identify effective strategies.

5. **Visual Documentation**: Creates persistent visual records of testing results for later analysis.

This approach combines both qualitative and quantitative assessment methods to provide a holistic evaluation of the ROI detection system's performance, with particular focus on comparing human/agent placement against the optimal K-means solution.

## Usage

The script can be run from the command line with the following arguments:

```
python InteractiveBBoxPlacementVisualiser.py --dataset PATH_TO_DATASET --annotations PATH_TO_COCO_JSON [--crop_width WIDTH] [--crop_height HEIGHT] [--mode {manual,random,both}]
```

Required arguments:
- `--dataset`: Path to the dataset directory
- `--annotations`: Path to the COCO JSON annotation file

Optional arguments:
- `--crop_width`: Width of the crop window (default: 640)
- `--crop_height`: Height of the crop window (default: 640)
- `--mode`: Test mode - manual, random, or both (default: manual)