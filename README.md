# ROI Detection with Reinforcement Learning

This repository implements a Region of Interest (ROI) detection system using Reinforcement Learning (RL). The system automatically identifies and places optimal ROIs in images containing multiple objects, specifically focused on pedestrian detection.

## Project Overview

The core innovation of this project is using K-means clustering with an L∞ constraint to determine optimal ROI placement, combined with a reinforcement learning agent that learns to efficiently place ROIs to maximize coverage of annotated objects while minimizing computational resources.

## Key Features

- **K-means based optimal ROI discovery** algorithm that minimizes the number of ROIs while ensuring complete coverage
- **Custom Gym environment** for ROI placement with a multi-faceted reward system
- **PPO-based RL agent** with a specialized neural network that fuses image and state features
- **Transfer learning** using pretrained ResNet18 for efficient feature extraction
- **Interactive testing tool** for visual validation and performance analysis

## Repository Structure

```
├── agent.py                             # PPO agent implementation with custom NN architecture
├── main.py                              # Training script and entry point
├── ROIDataset.py                        # Dataset class for loading images and annotations
├── ROIDetectionEnv.py                   # Custom Gym environment for ROI detection
├── opt_roi.py                           # Standalone K-means ROI optimization and visualization
├── InterractveBBoxPlacementVisualiser.py # Interactive testing and visualization tool
└── docs/                                # Detailed documentation
    ├── project_overview.md              # Overview of the entire project
    ├── roi_dataset_doc.md               # Documentation for ROIDataset.py
    ├── roi_detection_env_doc.md         # Documentation for ROIDetectionEnv.py
    ├── agent_doc.md                     # Documentation for agent.py
    ├── main_doc.md                      # Documentation for main.py
    ├── opt_roi_doc.md                   # Documentation for opt_roi.py
    ├── interactive_visualizer_doc.md    # Documentation for the visualizer
    └── methodology_summary.md           # Comprehensive methodology explanation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/chatzianagnostis/ROI-on-HD-images-with-RL.git
cd ROI-on-HD-images-with-RL
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Agent

```bash
python main.py
```

This will start the training process using the configurations specified in `main.py`. By default, it will:
- Train for 1,000,000 timesteps
- Save checkpoints every 15,000 timesteps
- Save the final model as "final_kmeans_model.zip"

### Testing the Optimal ROI Algorithm

```bash
python opt_roi.py
```

This will run the standalone K-means based optimal ROI detection algorithm and visualize the results.

### Interactive Testing

```bash
python InterractveBBoxPlacementVisualiser.py --dataset PATH_TO_DATASET --annotations PATH_TO_COCO_JSON
```

Required arguments:
- `--dataset`: Path to the dataset directory
- `--annotations`: Path to the COCO JSON annotation file

Optional arguments:
- `--crop_width`: Width of the crop window (default: 640)
- `--crop_height`: Height of the crop window (default: 640)
- `--mode`: Test mode - manual, random, or both (default: manual)

## Methodology

The methodology combines:

1. **K-means clustering** to determine optimal ROI placement
2. **Reinforcement learning** to train an agent to place ROIs efficiently
3. **Custom reward system** balancing coverage, efficiency, and placement quality
4. **Feature fusion neural network** processing both visual and state information

For a detailed explanation of the methodology, see [docs/methodology_summary.md](docs/methodology_summary.md).

## Applications

This system is designed for scenarios where computational resources are limited and detection needs to be focused on specific regions rather than processing entire high-resolution images:

- Urban surveillance with high-resolution cameras
- Autonomous vehicles sensing
- Drone-based monitoring
- Industrial inspection systems

## Requirements

- Python 3.7+
- PyTorch 1.7+
- Stable-Baselines3
- OpenCV
- NumPy
- Gymnasium
- scikit-learn

A full list of dependencies can be found in `requirements.txt`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@misc{roi-detection-rl,
  author = {Your Name},
  title = {ROI Detection with Reinforcement Learning},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/roi-detection-rl}}
}
```

## Acknowledgements

- The project uses the Stable-Baselines3 implementation of PPO
- ResNet18 pretrained weights are from torchvision
- Thanks to the developers of OpenCV, PyTorch, and scikit-learn
