# ROI Detection with Reinforcement Learning - Project Overview

## Introduction

This project implements a Region of Interest (ROI) detection system using Reinforcement Learning (RL). The system is designed to automatically identify and place optimal ROIs in images containing multiple objects, particularly focused on pedestrian detection.

## Project Components

The project consists of several interconnected Python modules:

1. **ROIDataset.py**: Handles loading and preprocessing image data and annotations from various formats (YOLO, COCO).

2. **ROIDetectionEnv.py**: Implements a custom Gym environment for the ROI detection task, using a K-means clustering approach to determine optimal ROI placements.

3. **agent.py**: Defines a Proximal Policy Optimization (PPO) agent with a custom neural network architecture that combines CNN features from images with bounding box state information.

4. **main.py**: Contains the training loop and entry point for the application.

5. **opt_roi.py**: Implements and visualizes the optimal ROI detection algorithm based on K-means clustering.

6. **InteractiveBBoxPlacementVisualiser.py**: Provides a visual interface for testing and interacting with the environment.

## Methodology

### 1. Data Handling

The system uses a dataset of images with pedestrian annotations, typically in COCO format. The annotations include bounding boxes for pedestrians in each image. The `ROIDataset` class handles loading these images and annotations, resizing them to a standard size (640x640), and providing them to the RL environment.

### 2. Optimal ROI Determination

The core methodology uses K-means clustering to determine the optimal number and placement of ROIs. The algorithm:
- Collects the centers of all annotated objects
- Searches for the minimal number of clusters (k) where all annotations within each cluster can be covered by a single ROI
- Places ROIs centered on the cluster centers

This approach ensures minimum ROIs while maintaining complete coverage of all annotations.

### 3. Reinforcement Learning Environment

The environment allows an agent to learn how to place ROIs effectively:
- **State**: Current image and the state of already placed bounding boxes
- **Actions**: Move the current bounding box (up, down, left, right), place a box, remove the last placed box, or end the episode
- **Reward**: Based on annotation coverage, match with optimal ROIs, efficiency of ROI count, and penalties for overlapping ROIs

### 4. Neural Network Architecture

The agent uses a custom neural network combining:
- A ResNet18 backbone (with early layers frozen) for image feature extraction
- A separate network for processing the bounding box state
- Combined features processed through fully-connected layers to predict actions

### 5. Training Process

The training process uses PPO (Proximal Policy Optimization) with the following characteristics:
- Batch size: 64
- Learning steps per update: 512
- Multiple epochs per update: 10
- Standard discount factors and advantage estimation
- Periodic model checkpoints

## Applications

This system is designed for scenarios where computational resources are limited and detection needs to be focused on specific regions rather than processing entire high-resolution images. Potential applications include:

- Urban surveillance with high-resolution cameras
- Autonomous vehicles sensing
- Drone-based monitoring
- Industrial inspection systems

## Performance Metrics

The system's performance is evaluated based on:

1. Coverage score: Percentage of annotations properly covered by the placed ROIs
2. ROI matching score: How well the placed ROIs match the optimal ROIs (based on IoU)
3. Efficiency score: How close the number of placed ROIs is to the optimal number
4. Overlap penalty: Penalty for excessive overlap between ROIs

These metrics combine to provide a comprehensive evaluation of the ROI placement strategy.
