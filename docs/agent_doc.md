# agent.py - Detailed Documentation

## Overview
`agent.py` implements a deep reinforcement learning agent using Proximal Policy Optimization (PPO) for the ROI detection task. The agent uses a custom neural network architecture that combines CNN features extracted from images with bounding box state information.

## Class: ROIFeatureExtractor

### Initialization
```python
def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
```

- Initialize the feature extractor with:
  - `observation_space`: Dict space containing image and bbox_state
  - `features_dim`: Dimension of output features (default: 256)
- Load ResNet18 with pretrained weights and remove final FC layer
- Freeze most layers to speed up training (only last few layers remain trainable)
- Create network for bbox state processing:
  - Flatten and transform to 64-dimensional vector
- Create combined layer for feature fusion:
  - Concatenate image (512) and bbox (64) features
  - Transform to specified feature dimension (256)

### Forward Pass
```python
def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
```

- Process observations through the network:
  - Extract and normalize image from observations
  - Process through CNN (ResNet18) and flatten
  - Process bbox state through bbox encoder
  - Concatenate features and return combined representation

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

- Initialize the ROI agent with:
  - `env`: ROIDetectionEnv instance
  - `model_dir`: Directory for saving models
  - `log_dir`: Directory for logs
  - `tensorboard_log`: Directory for tensorboard logs
  - `learning_rate`: Learning rate for optimizer
- Configure PPO hyperparameters optimized for K-means environment:
  - n_steps: 2048 (steps per update)
  - batch_size: 64sssss
  - n_epochs: 10 (optimization epochs per update)
  - gamma: 0.995 (discount factor)
  - gae_lambda: 0.95 (advantage estimation parameter)
  - ent_coef: 1e-3 (entropy coefficient)
  - clip_range: 0.2 (PPO clipping parameter)
  - Custom feature extractor: ROIFeatureExtractor
- Initialize PPO model with parameters

### Training Method
```python
def train(self, total_timesteps=1_000_000, callback=None):
```

- Train the agent for specified number of timesteps
- Support callbacks for monitoring/checkpointing

### Model Saving and Loading
```python
def save_model(self, name="roi_kmeans_agent"):
def load_model(self, name="roi_kmeans_agent"):
```

- Save trained model to specified path
- Load model from specified path

### Prediction Method
```python
def predict(self, observation, deterministic=True):
```

- Make a prediction with the model
- Return action and state

## Key Innovations

1. **Custom Feature Extractor with ResNet18**:
   - Uses pretrained ResNet18 backbone for efficient image feature extraction
   - Freezes early layers to speed up training and prevent overfitting
   - Processes bbox state separately and combines with image features
   - Feature fusion enables context-aware decision making

2. **PPO Configuration Optimized for K-means Environment**:
   - Increased n_steps (2048) for more stable policy updates
   - Reduced batch size (32) for better generalization
   - Higher discount factor (0.995) for long-term planning
   - Lower entropy coefficient (1e-3) for more focused exploitation
   - Balanced advantage estimation (gae_lambda=0.95)

3. **Parameter Efficiency**:
   - Transfer learning with frozen early layers
   - Compact state processing network
   - Shared feature extraction for actor and critic

4. **Training Workflow**:
   - Checkpoint callback for periodic model saving
   - Tensorboard integration for monitoring
   - Configurable training parameters

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

## Methodology

The agent implements a deep reinforcement learning approach with several methodological considerations:

1. **Transfer Learning**: Utilizes pretrained ResNet18 to leverage pre-learned visual features.

2. **Feature Fusion**: Combines visual information with state information for context-aware decision making.

3. **Parameter Efficiency**: Freezes early CNN layers to reduce trainable parameters and speed up learning.

4. **Stable Policy Updates**: Uses PPO's clipping mechanism to prevent destructive policy updates.

5. **Exploration-Exploitation Balance**: Configures entropy coefficient to maintain appropriate exploration.

6. **Advantage Estimation**: Uses Generalized Advantage Estimation (GAE) for more stable learning signals.

This combination of techniques allows the agent to efficiently learn the ROI placement task, adapting to different images and object distributions while maintaining stable training dynamics.