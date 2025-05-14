"""
Region of Interest Dataset for Computer Vision and Reinforcement Learning.

This module implements a dataset class for loading and preprocessing images with
region of interest (ROI) annotations. It supports multiple annotation formats
including YOLO and COCO, and provides iteration capabilities for use in training.
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path
import random
from typing import Dict, List, Tuple, Optional, Generator, Any, Iterator


class ROIDataset:
    """
    Dataset class for ROI detection that handles loading images and labels.
    
    This class provides functionality to:
    - Load images and their corresponding annotations
    - Convert between different annotation formats (YOLO, COCO)
    - Resize images and scale annotations accordingly
    - Iterate through the dataset with optional shuffling
    
    It can be used as an iterator to efficiently process datasets of any size.
    """

    def __init__(
        self, 
        dataset_path: str, 
        image_size: Tuple[int, int] = (640, 640),
        annotations_format: str = "yolo",
        coco_json_path: Optional[str] = None,
        shuffle: bool = True,
        include_original: bool = True
    ):
        """
        Initialize the ROI dataset.
        
        Args:
            dataset_path: Path to the dataset directory containing images
            image_size: Target size for resized images (width, height)
            annotations_format: Format of annotations ('yolo', 'coco')
            coco_json_path: Optional path to a centralized COCO JSON file
            shuffle: Whether to shuffle the dataset when iterating
            include_original: Whether to keep original images in returned samples
        """
        # Store initialization parameters
        self.dataset_path = Path(dataset_path)
        self.image_size = image_size
        self.annotations_format = annotations_format.lower()
        self.coco_json_path = Path(coco_json_path) if coco_json_path else None
        self.shuffle = shuffle
        self.include_original = include_original
        
        # COCO specific data structures
        self.coco_data = None
        self.coco_image_map = {}
        
        # Load COCO annotations if using centralized JSON file
        if self.annotations_format == "coco" and self.coco_json_path:
            self._load_coco_annotations()
        
        # Find and prepare all valid samples (image-annotation pairs)
        self.samples = self._find_samples()
        
        # Shuffle if requested
        if shuffle:
            random.shuffle(self.samples)
            
        # Initialize iterator index
        self.current_idx = 0
    
    def _load_coco_annotations(self) -> None:
        """
        Load COCO annotations from a centralized JSON file.
        
        This method:
        1. Reads the COCO JSON file
        2. Creates a mapping from image ID to filename
        3. Creates a mapping from filename to annotations
        
        Errors are handled gracefully with appropriate error messages.
        """
        try:
            # Load JSON data
            with open(self.coco_json_path, 'r') as f:
                self.coco_data = json.load(f)
            
            # Map image IDs to filenames
            image_id_to_filename = {}
            for image in self.coco_data.get('images', []):
                image_id_to_filename[image['id']] = image['file_name']
            
            # Map filenames to their annotations
            for annotation in self.coco_data.get('annotations', []):
                image_id = annotation['image_id']
                if image_id in image_id_to_filename:
                    filename = image_id_to_filename[image_id]
                    if filename not in self.coco_image_map:
                        self.coco_image_map[filename] = []
                    self.coco_image_map[filename].append(annotation)
            
            print(f"Loaded {len(self.coco_image_map)} images with annotations from COCO JSON")
        except Exception as e:
            print(f"Error loading COCO JSON file {self.coco_json_path}: {e}")
            self.coco_data = None
            self.coco_image_map = {}
            
    def _find_samples(self) -> List[Dict[str, Any]]:
        """
        Find all valid image-annotation pairs in the dataset directory.
        
        This method:
        1. Searches for images with supported extensions
        2. Locates corresponding annotation files based on format
        3. Creates sample entries with paths to both
        
        Returns:
            List of dictionaries, each containing paths to an image and its annotation
        """
        samples = []
        # Supported image file extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        # Search recursively through all subdirectories
        for img_path in self.dataset_path.glob('**/*'):
            if img_path.suffix.lower() in image_extensions:
                # Determine annotation path based on format
                ann_path = None
                
                if self.annotations_format == "yolo":
                    # YOLO format: same name but .txt extension in same folder
                    ann_path = img_path.with_suffix('.txt')
                    
                elif self.annotations_format == "coco":
                    if self.coco_json_path:
                        # Using a centralized COCO JSON file
                        filename = img_path.name
                        if filename in self.coco_image_map:
                            # Mark that this image has annotations in the centralized file
                            ann_path = self.coco_json_path
                    else:
                        # Individual JSON files in annotations subdirectory
                        ann_path = self.dataset_path / "annotations" / f"{img_path.stem}.json"
                        
                else:
                    # Default fallback: look for same name with .txt extension
                    ann_path = img_path.with_suffix('.txt')
                
                # Only add as sample if annotations exist
                has_annotations = (
                    ann_path and 
                    (ann_path.exists() or 
                    (self.annotations_format == "coco" and 
                     self.coco_json_path and 
                     img_path.name in self.coco_image_map))
                )
                
                if has_annotations:
                    samples.append({
                        'image_path': str(img_path),
                        'annotation_path': str(ann_path),
                        'image_filename': img_path.name  # Store filename for COCO lookup
                    })
        
        print(f"Found {len(samples)} valid image-annotation pairs")
        return samples
    
    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        
        Returns:
            int: Total number of valid samples
        """
        return len(self.samples)
    
    def __iter__(self) -> 'ROIDataset':
        """
        Make the dataset iterable.
        
        This method resets the iterator and optionally shuffles the dataset.
        
        Returns:
            self: The dataset object itself as an iterator
        """
        self.current_idx = 0
        if self.shuffle:
            random.shuffle(self.samples)
        return self
    
    def __next__(self) -> Dict[str, Any]:
        """
        Get the next sample when iterating through the dataset.
        
        Returns:
            Dict: A processed sample containing the image and its annotations
            
        Raises:
            StopIteration: When all samples have been processed
        """
        if self.current_idx >= len(self.samples):
            raise StopIteration
            
        sample = self.get_item(self.current_idx)
        self.current_idx += 1
        return sample
    
    def get_item(self, idx: int) -> Dict[str, Any]:
        """
        Get a specific sample by index.
        
        This method:
        1. Loads the image
        2. Resizes it to the target size
        3. Loads and scales annotations
        4. Combines everything into a sample dictionary
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dict: Processed sample containing the image and annotations
            
        Raises:
            IndexError: If the index is out of range
            ValueError: If the image cannot be loaded
        """
        # Check for valid index
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.samples)} samples")
            
        sample_info = self.samples[idx]
        
        # Load image
        image_path = sample_info['image_path']
        original_image = cv2.imread(image_path)
        
        if original_image is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        # Get original dimensions for scaling annotations
        original_height, original_width = original_image.shape[:2]
        
        # Resize image for the environment
        resized_image = cv2.resize(original_image, self.image_size)
        
        # Calculate scale factors
        width_scale = self.image_size[0] / original_width
        height_scale = self.image_size[1] / original_height
        
        # Load annotations
        annotations = self._load_annotations(
            sample_info['annotation_path'], 
            original_width, 
            original_height,
            width_scale,
            height_scale,
            sample_info.get('image_filename')  # Pass filename for COCO lookup
        )
        
        # Prepare the result dictionary
        result = {
            'image_id': os.path.basename(image_path),
            'resized_image': resized_image,
            'annotations': annotations,
            'scale_factors': (width_scale, height_scale),
            'original_size': (original_width, original_height)
        }
        
        # Optionally include original image
        if self.include_original:
            result['original_image'] = original_image
            
        return result
        
    def _load_annotations(
        self, 
        annotation_path: str, 
        original_width: int, 
        original_height: int,
        width_scale: float,
        height_scale: float,
        image_filename: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Load annotations based on format and convert to standardized format.
        
        This method handles different annotation formats and scales them
        to match the resized image dimensions.
        
        Args:
            annotation_path: Path to the annotation file
            original_width: Width of the original image
            original_height: Height of the original image
            width_scale: Scale factor for width (resized/original)
            height_scale: Scale factor for height (resized/original)
            image_filename: Filename for COCO lookup (if using centralized JSON)
            
        Returns:
            List of dictionaries, each containing a standardized annotation
        """
        annotations = []
        
        if self.annotations_format == "yolo":
            # Process YOLO format annotations
            self._load_yolo_annotations(
                annotation_path,
                original_width,
                original_height,
                width_scale,
                height_scale,
                annotations
            )
                
        elif self.annotations_format == "coco":
            # Process COCO format annotations
            self._load_coco_format_annotations(
                annotation_path,
                width_scale,
                height_scale,
                image_filename,
                annotations
            )
        
        return annotations
    
    def _load_yolo_annotations(
        self,
        annotation_path: str,
        original_width: int,
        original_height: int,
        width_scale: float,
        height_scale: float,
        annotations: List[Dict[str, Any]]
    ) -> None:
        """
        Process YOLO format annotations and add them to the annotations list.
        
        YOLO format: class x_center y_center width height (normalized 0-1)
        
        Args:
            annotation_path: Path to the YOLO annotation file
            original_width: Width of the original image
            original_height: Height of the original image
            width_scale: Scale factor for width
            height_scale: Scale factor for height
            annotations: List to append processed annotations to
        """
        try:
            with open(annotation_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        # YOLO format is normalized [0-1] and centered
                        x_center = float(parts[1]) 
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Convert to top-left coordinates and absolute dimensions
                        x1 = (x_center - width/2) * original_width
                        y1 = (y_center - height/2) * original_height
                        w = width * original_width
                        h = height * original_height
                        
                        # Scale to resized image
                        x1_scaled = x1 * width_scale
                        y1_scaled = y1 * height_scale
                        w_scaled = w * width_scale
                        h_scaled = h * height_scale
                        
                        annotations.append({
                            'category_id': class_id,
                            'bbox_orig': [x1, y1, w, h],  # [x, y, width, height] in original image
                            'bbox': [x1_scaled, y1_scaled, w_scaled, h_scaled]  # in resized image
                        })
        except Exception as e:
            print(f"Error loading YOLO annotation {annotation_path}: {e}")
    
    def _load_coco_format_annotations(
        self,
        annotation_path: str,
        width_scale: float,
        height_scale: float,
        image_filename: Optional[str],
        annotations: List[Dict[str, Any]]
    ) -> None:
        """
        Process COCO format annotations and add them to the annotations list.
        
        COCO format: [x, y, width, height] in absolute pixels
        
        Args:
            annotation_path: Path to the COCO annotation file (or centralized JSON)
            width_scale: Scale factor for width
            height_scale: Scale factor for height
            image_filename: Filename for lookup in centralized COCO JSON
            annotations: List to append processed annotations to
        """
        if self.coco_json_path and image_filename and image_filename in self.coco_image_map:
            # Get annotations from the centralized COCO JSON
            self._process_centralized_coco_annotations(
                image_filename,
                width_scale,
                height_scale,
                annotations
            )
        else:
            # Process individual COCO JSON file
            self._process_individual_coco_file(
                annotation_path,
                width_scale,
                height_scale,
                annotations
            )
    
    def _process_centralized_coco_annotations(
        self,
        image_filename: str,
        width_scale: float,
        height_scale: float,
        annotations: List[Dict[str, Any]]
    ) -> None:
        """
        Process annotations from a centralized COCO JSON file.
        
        Args:
            image_filename: Filename to lookup in the COCO image map
            width_scale: Scale factor for width
            height_scale: Scale factor for height
            annotations: List to append processed annotations to
        """
        for ann in self.coco_image_map[image_filename]:
            if 'bbox' in ann and 'category_id' in ann:
                # COCO format is [x, y, width, height] in absolute pixels
                x, y, w, h = ann['bbox']
                
                # Scale to resized image
                x_scaled = x * width_scale
                y_scaled = y * height_scale
                w_scaled = w * width_scale
                h_scaled = h * height_scale
                
                annotations.append({
                    'category_id': ann['category_id'],
                    'bbox_orig': [x, y, w, h],  # in original image
                    'bbox': [x_scaled, y_scaled, w_scaled, h_scaled],  # in resized image
                    'score': ann.get('score', 1.0)
                })
    
    def _process_individual_coco_file(
        self,
        annotation_path: str,
        width_scale: float,
        height_scale: float,
        annotations: List[Dict[str, Any]]
    ) -> None:
        """
        Process annotations from an individual COCO JSON file.
        
        Args:
            annotation_path: Path to the individual COCO JSON file
            width_scale: Scale factor for width
            height_scale: Scale factor for height
            annotations: List to append processed annotations to
        """
        try:
            with open(annotation_path, 'r') as f:
                ann_data = json.load(f)
                
            for ann in ann_data.get('annotations', []):
                if 'bbox' in ann and 'category_id' in ann:
                    # COCO format is [x, y, width, height] in absolute pixels
                    x, y, w, h = ann['bbox']
                    
                    # Scale to resized image
                    x_scaled = x * width_scale
                    y_scaled = y * height_scale
                    w_scaled = w * width_scale
                    h_scaled = h * height_scale
                    
                    annotations.append({
                        'category_id': ann['category_id'],
                        'bbox_orig': [x, y, w, h],  # in original image
                        'bbox': [x_scaled, y_scaled, w_scaled, h_scaled],  # in resized image
                        'score': ann.get('score', 1.0)
                    })
        except Exception as e:
            print(f"Error loading individual COCO annotation {annotation_path}: {e}")
    
    def get_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        Get a batch of samples from the dataset.
        
        This method gets the next batch_size samples from the dataset,
        restarting iteration if the end is reached.
        
        Args:
            batch_size: Number of samples to retrieve
            
        Returns:
            List of processed samples
        """
        batch = []
        for _ in range(batch_size):
            try:
                batch.append(next(self))
            except StopIteration:
                # Reset iterator if we reach the end
                self.__iter__()
                if not batch:  # If we couldn't get any samples, try again
                    return self.get_batch(batch_size)
                break
                
        return batch