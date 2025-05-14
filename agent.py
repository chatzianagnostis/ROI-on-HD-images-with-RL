"""
Region of Interest Detection Agent using Maskable PPO.

This module implements a reinforcement learning agent and feature extractor
for training models to detect regions of interest in images.
"""

import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from gym import spaces

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback


class ROIFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for the ROI Detection environment.
    
    This class uses a pre-trained ResNet18 with frozen early layers to efficiently
    extract features from images, combined with a separate network to process
    bounding box states. The outputs are concatenated and passed through a
    final linear layer to produce a unified feature representation.
    """
    
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        """
        Initialize the feature extractor.
        
        Args:
            observation_space: The observation space from the environment
            features_dim: Dimension of the output feature vector
        """
        super(ROIFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # ----- Image Processing Network -----
        # Load ResNet18 and remove final FC layer
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Keep everything except the final fully connected layer
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        
        # Freeze early layers to speed up training and prevent overfitting
        # Only fine-tune the last few layers that are more task-specific
        num_params_to_unfreeze = 10  # Number of parameters to unfreeze
        
        # A more robust way to selectively freeze layers:
        for name, param in self.cnn.named_parameters():
            # Freeze everything except the last layer (layer4 in ResNet18)
            if not name.startswith("7."):  # layer4 is at index 7 in ResNet18
                param.requires_grad = False
        
        # ----- Bounding Box State Processing Network -----
        # Calculate input dimension from the observation space
        bbox_input_dim = np.prod(observation_space.spaces['bbox_state'].shape)
        
        self.bbox_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(bbox_input_dim, 64),
            nn.ReLU()
        )
        
        # ----- Combined Features Network -----
        # ResNet18 outputs 512-dimensional vectors before the FC layer
        self.combined_layer = nn.Sequential(
            nn.Linear(512 + 64, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process observations through the feature extractor.
        
        Args:
            observations: Dictionary containing 'image' and 'bbox_state'
            
        Returns:
            torch.Tensor: Extracted features of dimension features_dim
        """
        # ----- Process Image -----
        image = observations['image']  # Shape: (N, H, W, C) or (N, C, H, W)
        
        # Normalize images to [0, 1] range
        image = image.float() / 255.0
        
        # Extract image features
        image_features = self.cnn(image)
        image_features = torch.flatten(image_features, start_dim=1)
        
        # ----- Process Bounding Box State -----
        bbox_state = observations['bbox_state'].float()
        bbox_features = self.bbox_encoder(bbox_state)
        
        # ----- Combine Features -----
        combined = torch.cat([image_features, bbox_features], dim=1)
        return self.combined_layer(combined)


class ROIAgent:
    """
    Agent for ROI Detection using Maskable PPO.
    
    This class handles the training, saving, loading, and inference of a
    reinforcement learning agent for the ROI detection task. It uses MaskablePPO
    which supports action masking to ensure valid actions in the environment.
    """
    
    def __init__(
        self,
        env,
        model_dir: str = "models",
        log_dir: str = "logs",
        tensorboard_log: Optional[str] = None,
        learning_rate: float = 3e-4
    ):
        """
        Initialize the ROI Agent.
        
        Args:
            env: The gym environment
            model_dir: Directory to save models
            log_dir: Directory to save logs
            tensorboard_log: Directory for tensorboard logs (defaults to log_dir)
            learning_rate: Learning rate for the optimizer
        """
        # Create directories if they don't exist
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Set tensorboard log directory
        if tensorboard_log is None:
            tensorboard_log = log_dir
        else:
            os.makedirs(tensorboard_log, exist_ok=True)

        # Store environment and directories
        self.env = env
        self.model_dir = model_dir
        self.log_dir = log_dir
        
        # Configure MaskablePPO hyperparameters
        ppo_params = {
            # Core components
            "policy": MaskableActorCriticPolicy,
            "env": env,
            "learning_rate": learning_rate,
            
            # Sample collection parameters
            "n_steps": 1024,
            "batch_size": 64,
            
            # Training parameters
            "n_epochs": 10,
            "gamma": 0.995,
            "gae_lambda": 0.995,
            "ent_coef": 0.0005,
            "clip_range": 0.2,
            "vf_coef": 0.005,
            "max_grad_norm": 0.5,
            "normalize_advantage": True,
            
            # Miscellaneous
            "verbose": 2,
            "tensorboard_log": tensorboard_log,
            
            # Custom feature extractor
            "policy_kwargs": {
                "features_extractor_class": ROIFeatureExtractor,
                "features_extractor_kwargs": {"features_dim": 256},
            }
        }

        # Initialize the MaskablePPO model
        self.model = MaskablePPO(**ppo_params)

    def train(self, total_timesteps: int = 1_000_000, callback: Optional[BaseCallback] = None) -> None:
        """
        Train the agent for a specified number of timesteps.
        
        Args:
            total_timesteps: Total number of environment timesteps to train for
            callback: Optional callback(s) to use during training for monitoring,
                      evaluation, or early stopping
        """
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def save_model(self, name: str = "roi_maskable_agent") -> None:
        """
        Save the trained model to disk.
        
        Args:
            name: Base name for the saved model file
        """
        path = os.path.join(self.model_dir, name)
        self.model.save(path)
        print(f"Model saved to {path}.zip")

    def load_model(self, name: str = "roi_maskable_agent") -> None:
        """
        Load a trained model from disk.
        
        Args:
            name: Base name of the model file to load
        """
        path = os.path.join(self.model_dir, name)
        self.model = MaskablePPO.load(path, env=self.env)
        print(f"Model loaded from {path}.zip")
        
    def predict(self, observation: Dict[str, np.ndarray], deterministic: bool = True) -> np.ndarray:
        """
        Make a prediction with the model, applying action masks.
        
        This method retrieves the current action mask from the environment
        and uses it to ensure only valid actions are selected.
        
        Args:
            observation: Environment observation
            deterministic: Whether to use deterministic actions (True for evaluation)
            
        Returns:
            np.ndarray: Selected action
        """
        # Get action mask from environment if available
        action_mask = None
        
        if hasattr(self.env, 'action_masks') and callable(self.env.action_masks):
            action_mask = self.env.action_masks()
        else:
            print("Warning: Could not get action_masks. Prediction might include invalid actions.")

        # Make prediction using the masked policy
        action, _states = self.model.predict(
            observation,
            deterministic=deterministic,
            action_masks=action_mask
        )
        
        return action