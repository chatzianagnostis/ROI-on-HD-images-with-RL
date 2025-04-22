# agent.py - Detailed Documentation

## Overview
`agent.py` implements a deep reinforcement learning agent using Proximal Policy Optimization (PPO) for the ROI detection task. The agent uses a custom neural network architecture that combines CNN features extracted from images with bounding box state information.

## Class: ROIFeatureExtractor

### Initialization
```python
def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
```

- **Lines 11-37**: Initialize the feature extractor
  - **Line 12**: Call parent class initializer
  - **Lines 15-16**: Load ResNet18 and remove final fully connected layer
    - Uses pretrained weights from torchvision models
    - Removes the classification head
  - **Lines 19-20**: Freeze most of the ResNet18 layers
    - Keeps only the last few layers trainable
    - Speeds up training by reducing parameters
  - **Lines 23-27**: Create network for processing bbox state
    - Flattens the state tensor
    - Linear layer to reduce to 64 dimensions
    - ReLU activation
  - **Lines 30-33**: Create combined layer for features
    - Linear layer to integrate image and bbox features
    - Output dimension specified by `features_dim` (default: 256)
    - ReLU activation

### Forward Pass
```python
def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
```

- **Lines 39-59**: Process observations through the network
  - **Lines 40-41**: Extract image from observations
  - **Lines 44-45**: Normalize pixel values to [0, 1]
  - **Lines 48-49**: Process through CNN (ResNet18)
    - Extract image features
    - Flatten to 1D vector
  - **Lines 52-57**: Process bbox state
    - Extract bbox state tensor
    - Handle different batch dimensions
    - Process through bbox encoder network
  - **Line 60**: Combine features and return processed representation

## Class: ROIAgent

### Initialization
```python
def __init__(
    self,
    env,
    model_dir="models",
    log_dir="logs",
    tensorboard_log="logs/tensorboard",
    learning_rate=3e-4
):
```

- **Lines 64-93**: Initialize the ROI agent
  - **Lines 74-76**: Create output directories
  - **Lines 79-92**: Configure PPO hyperparameters
    - Policy network: ActorCriticPolicy
    - Learning rate: 3e-4 (default)
    - n_steps: 512 (steps per update)
    - batch_size: 64
    - n_epochs: 10 (optimization epochs per update)
    - gamma: 0.99 (discount factor)
    - gae_lambda: 0.95 (advantage estimation parameter)
    - ent_coef: 0.01 (entropy coefficient)
    - clip_range: 0.2 (PPO clipping parameter)
    - Custom feature extractor: ROIFeatureExtractor
  - **Line 95**: Initialize PPO model with parameters
  - **Lines 97-99**: Store environment and directory paths

### Training Method
```python
def train(self, total_timesteps=1_000_000, callback=None):
```

- **Lines 101-109**: Train the agent
  - **Line 108**: Call PPO's learn method
  - Parameters:
    - `total_timesteps`: Number of timesteps to train for (default: 1,000,000)
    - `callback`: Optional callback(s) for monitoring/saving

### Model Saving and Loading
```python
def save_model(self, name="roi_kmeans_agent"):
def load_model(self, name="roi_kmeans_agent"):
```

- **Lines 111-114**: Save the trained model
  - Construct path from model directory and name
  - Call PPO's save method
- **Lines 116-118**: Load a trained model
  - Construct path from model directory and name
  - Load PPO model with the specified environment

### Prediction Method
```python
def predict(self, observation, deterministic=True):
```

- **Lines 120-122**: Make a prediction with the model
  - Call PPO's predict method
  - `deterministic`: Whether to use deterministic actions (default: True)
  - Returns action and state

## Key Features

1. **Custom Feature Extractor**:
   - Combines vision and state processing
   - Uses pretrained ResNet18 for efficient image feature extraction
   - Freezes early layers to speed up training
   - Processes bbox state separately and combines with image features

2. **PPO Implementation**:
   - Implements Proximal Policy Optimization algorithm
   - Uses actor-critic architecture
   - Configures appropriate hyperparameters
   - Supports tensorboard logging

3. **Model Management**:
   - Provides methods for saving and loading models
   - Creates organized directory structure
   - Supports named model variants

4. **Training Workflow**:
   - Encapsulates training process in a clean interface
   - Supports callbacks for monitoring and checkpointing
   - Configurable training duration

## Neural Network Architecture

The agent's neural network architecture consists of three main components:

1. **Image Feature Extractor**:
   - Based on ResNet18 with pretrained weights
   - Removes final classification layer
   - Freezes early layers to maintain low-level feature detection
   - Produces 512-dimensional feature vector

2. **Bounding Box State Processor**:
   - Flattens the state representation
   - Linear layer reduces dimension to 64
   - ReLU activation for non-linearity

3. **Feature Combiner**:
   - Concatenates the 512D image features with 64D bbox features
   - Linear layer reduces the combined 576D vector to 256D
   - ReLU activation for final features

This architecture enables the network to process visual information and maintain awareness of already placed ROIs, allowing for informed decisions on ROI placement.

## PPO Configuration

The Proximal Policy Optimization algorithm is configured with the following key parameters:

- **Learning rate**: 3e-4 (standard for Adam optimizer)
- **Steps per update**: 512 (collects 512 timesteps before each update)
- **Batch size**: 64 (processes 64 samples per optimization step)
- **Epochs per update**: 10 (performs 10 optimization iterations per update)
- **Discount factor (gamma)**: 0.99 (standard for long-term reward consideration)
- **GAE lambda**: 0.95 (controls trade-off between bias and variance in advantage estimation)
- **Entropy coefficient**: 0.01 (encourages exploration)
- **Clip range**: 0.2 (standard PPO clipping to prevent too large policy updates)

These parameters are optimized specifically for the K-means based environment to balance exploration, learning speed, and stability.

## Methodology

The agent implements a deep reinforcement learning approach with several methodological considerations:

1. **Transfer Learning**: Utilizes pretrained ResNet18 to leverage pre-learned visual features.

2. **Feature Fusion**: Combines visual information with state information for context-aware decision making.

3. **Parameter Efficiency**: Freezes early CNN layers to reduce trainable parameters and speed up learning.

4. **Stable Policy Updates**: Uses PPO's clipping mechanism to prevent destructive policy updates.

5. **Exploration-Exploitation Balance**: Configures entropy coefficient to maintain appropriate exploration.

6. **Advantage Estimation**: Uses Generalized Advantage Estimation (GAE) for more stable learning signals.

This combination of techniques allows the agent to efficiently learn the ROI placement task, adapting to different images and object distributions while maintaining stable training dynamics.
