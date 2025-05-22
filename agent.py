"""
Region of Interest Detection Agent using DINOv2.
"""

import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from gym import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback


class ROIFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for the ROI Detection environment.
    
    This class uses a pre-trained DINOv2-ViT-B/14 with frozen weights to efficiently
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
        # Load DINOv2-ViT-B/14 pretrained model
        print("Loading DINOv2-ViT-B/14...")
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True)
        
        # Freeze the ENTIRE DINOv2 - no fine tuning
        for param in self.dinov2.parameters():
            param.requires_grad = False
        
        self.dinov2.eval()
        
        # Register ImageNet normalization buffers for DINOv2
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # ----- Bounding Box State Processing Network -----
        # Calculate input dimension from the observation space
        bbox_input_dim = np.prod(observation_space.spaces['bbox_state'].shape)
        
        self.bbox_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(bbox_input_dim, 64),
            nn.ReLU()
        )
        
        # ----- Combined Features Network -----
        # DINOv2-ViT-B/14 outputs 768-dimensional vectors
        self.combined_layer = nn.Sequential(
            nn.Linear(768 + 64, features_dim),
            nn.ReLU()
        )
    
    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Preprocess image for DINOv2.
        Applies ImageNet normalization required by DINOv2.
        """
        # Normalize to [0, 1] range if needed
        if image.max() > 1.0:
            image = image.float() / 255.0

        # Resize to dimensions divisible by 14 (patch size)
        import torch.nn.functional as F
        image = F.interpolate(image, size=(644, 644), mode='bilinear', align_corners=False)
        
        # Apply ImageNet normalization (required for DINOv2)
        image = (image - self.mean) / self.std
        
        return image
        
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
        
        # Preprocess for DINOv2
        image = self.preprocess_image(image)
        
        # Extract image features using DINOv2
        with torch.no_grad():
            image_features = self.dinov2(image)  # Output: (N, 768)
        
        # ----- Process Bounding Box State -----
        bbox_state = observations['bbox_state'].float()
        bbox_features = self.bbox_encoder(bbox_state)
        
        # ----- Combine Features -----
        combined = torch.cat([image_features, bbox_features], dim=1)
        return self.combined_layer(combined)


class ROIAgent:
    """
    Agent for ROI Detection using standard PPO.
    
    This class handles the training, saving, loading, and inference of a
    reinforcement learning agent for the ROI detection task using standard PPO.
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

        net_arch = dict(pi=[1024, 512, 256], vf=[1024, 512, 256]) 
        
        # Configure PPO hyperparameters
        ppo_params = {
            # Core components
            "policy": ActorCriticPolicy,
            "env": env,
            "learning_rate": learning_rate,
            
            # Sample collection parameters
            "n_steps": 512,
            "batch_size": 16,
            
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
                "activation_fn": torch.nn.ReLU,    
                "optimizer_class": torch.optim.RAdam,
                "net_arch": net_arch,
            }
        }

        # Initialize the PPO model
        self.model = PPO(**ppo_params)

    def train(self, total_timesteps: int = 1_000_000, callback: Optional[BaseCallback] = None) -> None:
        """
        Train the agent for a specified number of timesteps.
        
        Args:
            total_timesteps: Total number of environment timesteps to train for
            callback: Optional callback(s) to use during training for monitoring,
                      evaluation, or early stopping
        """
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def save_model(self, name: str = "roi_ppo_agent") -> None:
        """
        Save the trained model to disk.
        
        Args:
            name: Base name for the saved model file
        """
        path = os.path.join(self.model_dir, name)
        self.model.save(path)
        print(f"Model saved to {path}.zip")

    def load_model(self, name: str = "roi_ppo_agent") -> None:
        """
        Load a trained model from disk.
        
        Args:
            name: Base name of the model file to load
        """
        path = os.path.join(self.model_dir, name)
        self.model = PPO.load(path, env=self.env)
        print(f"Model loaded from {path}.zip")
        
    def predict(self, observation: Dict[str, np.ndarray], deterministic: bool = True) -> np.ndarray:
        """
        Make a prediction with the model.
        
        Args:
            observation: Environment observation
            deterministic: Whether to use deterministic actions (True for evaluation)
            
        Returns:
            np.ndarray: Selected action
        """
        action, _states = self.model.predict(
            observation,
            deterministic=deterministic
        )
        
        return action