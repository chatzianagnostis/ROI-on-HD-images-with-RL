# ROIDataset.py - Detailed Documentation

## Overview
`ROIDataset.py` implements a dataset class for loading and preprocessing images and their annotations for the ROI detection task. It supports multiple annotation formats (YOLO, COCO) and provides a flexible, iterable interface for accessing the data.

## Class: ROIDataset

### Initialization
```python
def __init__(
    self, 
    dataset_path: str, 
    image_size: Tuple[int, int] = (640, 640),
    annotations_format: str = "yolo",
    coco_json_path: Optional[str] = None,
    shuffle: bool = True,
    include_original: bool = True
):
```

- **Lines 17-36**: Initialize dataset parameters
  - `dataset_path`: Path to the directory containing images
  - `image_size`: Target size for resizing images (width, height)
  - `annotations_format`: Format of annotations ("yolo", "coco")
  - `coco_json_path`: Path to a central COCO JSON file (optional)
  - `shuffle`: Whether to shuffle the dataset
  - `include_original`: Whether to keep original images in memory

- **Lines 37-40**: Initialize COCO-specific variables
  - `coco_data`: Will hold loaded COCO annotations
  - `coco_image_map`: Maps image filenames to their annotations

- **Lines 41-43**: Load COCO annotations if specified

- **Lines 46-49**: Find image-annotation pairs and initialize iteration counter
  - Calls `_find_samples()` to find valid pairs
  - Shuffles samples if requested
  - Sets `current_idx` to 0 for iteration

### COCO Annotation Loading
```python
def _load_coco_annotations(self):
```

- **Lines 51-76**: Load annotations from a single COCO JSON file
  - **Lines 53-55**: Load JSON data
  - **Lines 58-61**: Create mapping from image IDs to filenames
  - **Lines 64-72**: Create mapping from filenames to annotations
  - **Line 74**: Print loading status
  - **Lines 75-78**: Handle exceptions and reset data on error

### Finding Valid Samples
```python
def _find_samples(self) -> List[Dict[str, Any]]:
```

- **Lines 80-130**: Find all valid image-annotation pairs
  - **Lines 81-83**: Initialize variables
  - **Lines 85-128**: Iterate through all files in the dataset directory
    - **Lines 86-87**: Filter for image files
    - **Lines 90-106**: Determine the annotation path based on format
      - For YOLO: Look for .txt files with same name
      - For COCO with central JSON: Check if image has annotations in loaded data
      - For COCO with individual files: Look for JSON in annotations subdirectory
    - **Lines 109-117**: Add valid pairs to samples
  - **Line 129**: Return the list of samples with status message

### Iterator Methods
```python
def __len__(self) -> int:
def __iter__(self) -> 'ROIDataset':
def __next__(self) -> Dict[str, Any]:
```

- **Lines 132-134**: Return the number of samples
- **Lines 136-140**: Reset iteration
  - Reset index to 0
  - Reshuffle if enabled
  - Return self as iterator
- **Lines 142-148**: Get next sample for iteration
  - Check if there are more samples
  - Get the item at current index
  - Increment index
  - Return the sample

### Sample Retrieval
```python
def get_item(self, idx: int) -> Dict[str, Any]:
```

- **Lines 150-192**: Retrieve a specific sample by index
  - **Lines 151-153**: Check index validity
  - **Lines 156-160**: Load image and handle loading errors
  - **Lines 163-164**: Get original dimensions
  - **Lines 167-170**: Resize image and calculate scale factors
  - **Lines 173-179**: Load annotations with scaling applied
  - **Lines 181-190**: Prepare and return the sample dictionary
    - Include image ID, resized image, annotations, scaling factors
    - Optionally include original image

### Annotation Loading
```python
def _load_annotations(
    self, 
    annotation_path: str, 
    original_width: int, 
    original_height: int,
    width_scale: float,
    height_scale: float,
    image_filename: Optional[str] = None
) -> List[Dict[str, Any]]:
```

- **Lines 194-267**: Load and standardize annotations
  - **Line 197**: Initialize empty annotations list
  - **Lines 199-231**: Handle YOLO format
    - **Lines 202-230**: Parse each line of the annotation file
    - Convert normalized YOLO format to absolute coordinates
    - Scale coordinates to match the resized image
  - **Lines 233-265**: Handle COCO format
    - **Lines 234-253**: Process annotations from centralized COCO JSON if available
    - **Lines 254-264**: Process individual JSON files if not using central JSON
  - **Line 267**: Return the processed annotations

### Batch Retrieval
```python
def get_batch(self, batch_size: int) -> List[Dict[str, Any]]:
```

- **Lines 269-281**: Retrieve a batch of samples
  - **Line 270**: Initialize an empty batch
  - **Lines 271-278**: Try to add `batch_size` samples to the batch
  - **Line 280**: Return the batch

## Key Features

1. **Flexible Annotation Support**: Handles both YOLO and COCO annotation formats

2. **Efficient Preprocessing**: 
   - Resizes images to standardized dimensions
   - Scales annotations to match the resized images
   - Preprocesses data once during loading

3. **Iteration Support**:
   - Implements Python iterator protocol
   - Supports shuffling for randomized training
   - Provides batch retrieval for efficient training

4. **Robust Error Handling**:
   - Checks for file existence
   - Handles exceptions during loading
   - Validates annotations

5. **Data Conversion**:
   - Converts between different coordinate systems
   - Normalizes annotation formats
   - Tracks scale factors for later reference

## Methodology

The dataset class follows these methodological steps:

1. **Dataset Discovery**: Scan directories to find valid image-annotation pairs
2. **Format Adaptation**: Support multiple annotation formats through format-specific handlers
3. **Standardization**: Convert all annotations to a common format
4. **Preprocessing**: Resize images and scale annotations accordingly
5. **Batching**: Group samples for efficient processing

This approach ensures consistent data handling regardless of the original format, making the downstream processing more uniform and robust.
