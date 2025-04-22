# InteractiveBBoxPlacementVisualiser.py - Detailed Documentation

## Overview
`InteractiveBBoxPlacementVisualiser.py` implements an interactive testing tool for the ROI Detection Environment. It allows manual control or random testing of the environment, providing visual feedback and metrics on ROI placement performance.

## Class: ROIEnvTester

### Initialization
```python
def __init__(self, dataset_path, coco_json_path, crop_size=(640, 640)):
```

- **Lines 14-48**: Initialize the environment tester
  - **Lines 18-19**: Create output directory for results
  - **Lines 22-29**: Initialize dataset with specified parameters
    - Uses specified dataset and annotation paths
    - Sets image size to 640x640
    - Uses COCO annotation format
    - Enables shuffling
  - **Lines 32-36**: Initialize environment with K-means based ROI optimization
    - Uses the dataset created above
    - Sets crop size as specified
    - Sets longer time limit (300s) for testing
  - **Lines 39-45**: Initialize testing variables
    - Window name for display
    - Action and reward history for analysis
  - **Lines 47-48**: Print environment information

### Manual Episode Testing
```python
def run_manual_episode(self):
```

- **Lines 50-188**: Run a manual episode with user control
  - **Lines 51-62**: Print control instructions
    - W/Up: Move up
    - S/Down: Move down
    - A/Left: Move left
    - D/Right: Move right
    - Space: Place box
    - R: Remove last box
    - E: End episode
    - Q: Quit
    - K: Toggle optimal ROI visibility
  - **Lines 65-67**: Reset environment and initialize variables
  - **Lines 70-71**: Initialize toggle for showing optimal ROIs
  - **Lines 74-83**: Prepare and display initial frame
  - **Lines 85-140**: Main interaction loop
    - Wait for key press
    - Map key to action
    - Apply action to environment
    - Record action and reward
    - Update display
  - **Lines 142-187**: Handle episode ending
    - Print final statistics
    - Save final frame
    - Close windows

### Random Episode Testing
```python
def run_random_episode(self, num_steps=100, render=True):
```

- **Lines 190-266**: Run an episode with random actions
  - **Lines 191-205**: Initialize environment and display
  - **Lines 207-244**: Step loop
    - Generate random action with weighted probabilities
    - Apply action to environment
    - Record action and reward
    - Update display
  - **Lines 246-265**: Handle episode ending
    - Print final statistics
    - Save final frame
    - Close windows

### Action Analysis
```python
def analyze_action_distribution(self):
```

- **Lines 268-309**: Analyze distribution of actions and rewards
  - **Lines 269-271**: Check if actions were recorded
  - **Lines 273-279**: Print action distribution
    - Count occurrences of each action
    - Calculate percentages
  - **Lines 281-286**: Print reward statistics
    - Total, average, min, and max rewards
  - **Lines 289-300**: Group rewards by action
  - **Lines 302-309**: Print average reward by action

## Main Function and Entry Point

```python
def main():
```

- **Lines 311-333**: Parse arguments and run the tester
  - **Lines 312-322**: Set up argument parser
    - Required: dataset path and annotations path
    - Optional: crop dimensions, test mode
  - **Lines 324-327**: Create tester with specified parameters
  - **Lines 329-332**: Run tests based on selected mode
    - Manual: User-controlled testing
    - Random: Automated random action testing
    - Both: Run both types of tests
  - **Line 333**: Analyze action distribution after testing

```python
if __name__ == "__main__":
    main()
```

- **Lines 335-336**: Standard Python entry point

## Key Features

### 1. Interactive Testing
- **Manual Control**: Allows direct control of ROI placement
- **Key Mappings**: Simple keyboard controls for all actions
- **Visual Feedback**: Real-time display of environment state

### 2. Automated Testing
- **Random Action Generation**: Tests environment with weighted random actions
- **Configurable Steps**: Adjustable number of random steps
- **Optional Rendering**: Can run with or without visual display

### 3. Performance Analysis
- **Action Distribution**: Analyzes frequency of different actions
- **Reward Analysis**: Calculates statistics on rewards
- **Action-Reward Correlation**: Examines which actions yield higher rewards

### 4. Visualization
- **Color Coding**:
  - Yellow: Ground truth annotations
  - Red: Optimal ROIs from K-means
  - Green: User-placed ROIs
  - Blue: Current movable ROI
- **Toggle Functionality**: Can show/hide optimal ROIs
- **Result Saving**: Automatically saves final state images

## Testing Modes

### 1. Manual Mode
Allows a user to directly interact with the environment using keyboard controls. This mode is useful for:
- Understanding how the environment responds to actions
- Testing specific ROI placement strategies
- Visually comparing manual placement with K-means optimal placement

### 2. Random Mode
Automatically tests the environment with random actions. This mode is useful for:
- Quickly generating baseline performance metrics
- Testing environment stability
- Collecting action-reward statistics without user bias

## Action Weighting

When generating random actions, the actions are weighted with the following probabilities:
- Move Up: 20%
- Move Down: 20%
- Move Left: 20%
- Move Right: 20%
- Place Box: 10%
- Remove Box: 5%
- End Episode: 5%

This weighting prioritizes movement to explore the state space more thoroughly before committing to box placement.

## Methodology

The visualizer tool implements a comprehensive methodology for testing and evaluating the ROI Detection Environment:

1. **Interactive Validation**: Provides a means to manually verify environment behavior and visually assess the K-means algorithm's effectiveness.

2. **Comparative Analysis**: Enables direct comparison between manual ROI placement and the automated K-means approach.

3. **Quantitative Evaluation**: Collects and analyzes metrics to objectively measure performance.

4. **Statistical Analysis**: Examines patterns in actions and rewards to identify effective strategies.

5. **Visual Documentation**: Creates persistent visual records of testing results for later analysis.

This approach combines both qualitative and quantitative assessment methods to provide a holistic evaluation of the ROI detection system's performance.

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
