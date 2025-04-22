# opt_roi.py - Detailed Documentation

## Overview
`opt_roi.py` implements and visualizes the optimal ROI detection algorithm based on K-means clustering. This standalone script can be used to test and validate the K-means approach for determining optimal ROI placement.

## Functions

### Main Test Function
```python
def test_optimal_roi_detection():
```

- **Lines 4-41**: Initialize test environment and run visualization
  - **Lines 6-7**: Define paths to dataset and annotations
  - **Lines 10-23**: Initialize parameters and dataset
    - Set crop size to 640x640
    - Create ROIDataset with specified parameters
    - Ensure image size is 640x640
  - **Line 26**: Get a sample from the dataset
  - **Lines 29-31**: Calculate visual bbox size for resized image
  - **Lines 33-35**: Print configuration information
  - **Line 38**: Calculate optimal ROIs using K-means
  - **Line 41**: Visualize the results

### K-means ROI Calculation
```python
def calculate_optimal_rois(sample, roi_size):
```

- **Lines 46-81**: K-means-based ROI discovery under L∞ constraint
  - **Lines 46-51**: Function documentation
  - **Lines 52-55**: Extract required information from sample
  - **Lines 58-59**: Define helper to clamp values
  - **Lines 62-66**: Collect annotation centers as points
  - **Lines 69-70**: Set initial values for optimization
  - **Lines 71-95**: Iterative search for minimal k
    - Try k from 1 to number of annotations
    - For each k, fit K-means clustering
    - Check if all points in each cluster fit within a single ROI
    - If all fit, record k and centers and break
  - **Lines 98-105**: Build ROIs around cluster centers
    - Clamp centers to ensure ROIs stay within image
    - Convert to [x0, y0, width, height] format
  - **Line 107**: Return the list of ROIs

### Annotation Coverage Checking
```python
def is_annotation_covered(annotation, roi):
```

- **Lines 113-133**: Check if an annotation is fully covered by an ROI
  - **Line 114**: Extract bounding box from annotation
  - **Lines 117-121**: Convert annotation to corner format [x1, y1, x2, y2]
  - **Lines 123-127**: Extract ROI corners
  - **Lines 130-131**: Check if annotation is fully contained in ROI

### Visualization Function
```python
def visualize_results(sample, optimal_rois, bbox_size):
```

- **Lines 135-207**: Visualize annotations and optimal ROIs
  - **Line 136**: Copy the resized image
  - **Lines 140-146**: Draw ground truth annotations in yellow
  - **Lines 149-158**: Draw optimal ROIs in green with numbering
  - **Lines 161-169**: Calculate coverage statistics
    - Count annotations covered by at least one ROI
    - Calculate coverage percentage
  - **Lines 171-173**: Print coverage information
  - **Lines 176-181**: Add title with results information
  - **Lines 184-185**: Save the visualization
  - **Lines 188-191**: Show the image in a window

## Script Entry Point
```python
if __name__ == "__main__":
    test_optimal_roi_detection()
```

- **Lines 209-210**: Standard Python entry point
  - Runs the test and visualization when script is executed directly

## Key Components

### 1. K-means Clustering Algorithm
The core methodology is implemented in `calculate_optimal_rois`:

- **Center Collection**: Extracts the center point of each annotation
- **Minimal K Search**: Finds the smallest number of clusters that satisfy the constraint
- **L∞ Constraint**: Ensures all points in a cluster can be covered by a single ROI
- **ROI Construction**: Places ROIs centered on cluster centers

### 2. Evaluation Metrics
The script calculates and visualizes key evaluation metrics:

- **Coverage**: Percentage of annotations covered by the ROIs
- **Efficiency**: Average annotations per ROI
- **ROI Count**: Number of ROIs needed for coverage

### 3. Visualization
The visualization component provides clear visual feedback:

- **Color Coding**: 
  - Yellow: Ground truth annotations
  - Green: Optimal ROIs
- **Numbering**: Each ROI is numbered for reference
- **Statistics**: Overlaid title with coverage and efficiency metrics

## Algorithm Details

The optimal ROI calculation follows these steps:

1. **Extract Centers**: For each annotation, calculate its center point (x+w/2, y+h/2)
2. **Iterative Search**: Starting from k=1, incrementally increase k until a valid solution is found
3. **Validity Check**: For each k, evaluate if all points within each cluster satisfy:
   - max |Δx| ≤ half_width
   - max |Δy| ≤ half_height
   Where Δx and Δy are distances from points to the cluster center
4. **ROI Placement**: Create ROIs centered on each valid cluster center
5. **Boundary Handling**: Ensure ROIs stay within image boundaries by clamping center coordinates

This algorithm guarantees that:
- Each annotation is fully contained within at least one ROI
- The number of ROIs is minimized
- All ROIs remain within the image boundaries

## Methodology

The script implements a principled approach to ROI placement optimization:

1. **Problem Formulation**: Define the task as finding the minimum number of ROIs required to cover all annotations, where each ROI has a fixed size.

2. **Clustering Approach**: Use K-means clustering to group annotations based on spatial proximity.

3. **L∞ Constraint**: Apply a maximum deviation constraint to ensure all points in a cluster can be covered by a single ROI.

4. **Incremental Search**: Find the minimum k by iteratively testing increasing values.

5. **Visual Validation**: Provide visual confirmation of the algorithm's effectiveness.

This methodology provides a computationally efficient way to determine optimal ROI placement while ensuring complete coverage of all annotations.
