import os
import cv2
import numpy as np
import json
from pathlib import Path
import random
from typing import Dict, List, Tuple, Optional, Generator, Any

class ROIDataset:
    """
    Dataset class for ROI detection that handles loading images and labels.
    Can be used as a generator to iterate through the dataset.
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
            dataset_path: Path to the dataset directory
            image_size: Size to resize images to (width, height)
            annotations_format: Format of annotations ('yolo', 'coco', etc.)
            coco_json_path: Optional direct path to a COCO JSON file
            shuffle: Whether to shuffle the dataset
            include_original: Whether to keep original images in memory
        """
        self.dataset_path = Path(dataset_path)
        self.image_size = image_size
        self.annotations_format = annotations_format
        self.coco_json_path = Path(coco_json_path) if coco_json_path else None
        self.shuffle = shuffle
        self.include_original = include_original
        
        # Load COCO annotations from a single file if provided
        self.coco_data = None
        self.coco_image_map = {}
        if self.annotations_format == "coco" and self.coco_json_path:
            self._load_coco_annotations()
        
        # Find all images and their corresponding labels
        self.samples = self._find_samples()
        
        if shuffle:
            random.shuffle(self.samples)
            
        self.current_idx = 0
    
    def _load_coco_annotations(self):
        """Load COCO annotations from a single JSON file"""
        try:
            with open(self.coco_json_path, 'r') as f:
                self.coco_data = json.load(f)
            
            # Create a mapping from image_id to image filename
            image_id_to_filename = {}
            for image in self.coco_data.get('images', []):
                image_id_to_filename[image['id']] = image['file_name']
            
            # Create a mapping from image filename to annotations
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
        """Find all valid image-annotation pairs in the dataset directory"""
        samples = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for img_path in self.dataset_path.glob('**/*'):
            if img_path.suffix.lower() in image_extensions:
                # Handle different annotation formats
                ann_path = None
                
                if self.annotations_format == "yolo":
                    # YOLO format: same name but .txt extension in same folder
                    ann_path = img_path.with_suffix('.txt')
                elif self.annotations_format == "coco":
                    if self.coco_json_path:
                        # We're using a central COCO JSON file
                        # Check if this image has annotations in our loaded data
                        filename = img_path.name
                        if filename in self.coco_image_map:
                            # Use the JSON path as a marker that we have annotations
                            ann_path = self.coco_json_path
                    else:
                        # Individual JSON files in annotations subdirectory
                        ann_path = self.dataset_path / "annotations" / f"{img_path.stem}.json"
                else:
                    # Default to looking for same name with .txt extension
                    ann_path = img_path.with_suffix('.txt')
                
                # Only add as a sample if annotations exist
                if ann_path and (ann_path.exists() or 
                               (self.annotations_format == "coco" and self.coco_json_path)):
                    samples.append({
                        'image_path': str(img_path),
                        'annotation_path': str(ann_path),
                        'image_filename': img_path.name  # Store filename for COCO lookup
                    })
        
        print(f"Found {len(samples)} valid image-annotation pairs")
        return samples
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset"""
        return len(self.samples)
    
    def __iter__(self) -> 'ROIDataset':
        """Make the dataset iterable"""
        self.current_idx = 0
        if self.shuffle:
            random.shuffle(self.samples)
        return self
    
    def __next__(self) -> Dict[str, Any]:
        """Get the next sample when iterating"""
        if self.current_idx >= len(self.samples):
            raise StopIteration
            
        sample = self.get_item(self.current_idx)
        self.current_idx += 1
        return sample
    
    def get_item(self, idx: int) -> Dict[str, Any]:
        """Get a specific sample by index"""
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
        Returns list of dicts with 'bbox', 'category_id', etc.
        """
        annotations = []
        
        if self.annotations_format == "yolo":
            # YOLO format: class x_center y_center width height (normalized)
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
                            
                            # Convert to pixel coordinates in the original image
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
                
        elif self.annotations_format == "coco":
            if self.coco_json_path and image_filename and image_filename in self.coco_image_map:
                # Get annotations for this image from the loaded COCO data
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
                            'score': ann.get('score', 1.0) if 'score' in ann else 1.0
                        })
            else:
                # Traditional individual JSON files
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
                                'score': ann.get('score', 1.0) if 'score' in ann else 1.0
                            })
                except Exception as e:
                    print(f"Error loading individual COCO annotation {annotation_path}: {e}")
        
        return annotations
    
    def get_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Get a batch of samples"""
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