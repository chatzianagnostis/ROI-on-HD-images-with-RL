import os
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision import models
from gym import spaces
from typing import Dict, List, Optional

class ROIFeatureExtractor(BaseFeaturesExtractor):
    """Feature extractor for the ROI Detection environment using ResNet18"""
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super(ROIFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # Load ResNet18 and remove final FC layer
        self.cnn = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        
        # Freeze most layers to speed up training
        for param in list(self.cnn.parameters())[:-4]:
            param.requires_grad = False
            
        # Network for bbox state processing
        bbox_shape = observation_space.spaces['bbox_state'].shape
        self.bbox_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(bbox_shape[0] * bbox_shape[1], 64),
            nn.ReLU()
        )
        
        # Combine features
        self.combined_layer = nn.Sequential(
            nn.Linear(512 + 64, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Get the image from observations
        image = observations['image']
        
        # Normalize to [0, 1]
        image = image.float() / 255.0
        
        # Process through CNN
        image_features = self.cnn(image)
        image_features = torch.flatten(image_features, start_dim=1)
        
        # Process bbox state
        bbox_state = observations['bbox_state'].float()
        if len(bbox_state.shape) > 2:
            bbox_state = bbox_state.reshape(bbox_state.shape[0], -1)
        else:
            bbox_state = bbox_state.reshape(1, -1)
            
        bbox_features = self.bbox_encoder(bbox_state)
        
        # Combine features from both inputs
        combined = torch.cat([image_features, bbox_features], dim=1)
        return self.combined_layer(combined)

class ROIAgent:
    """Agent for ROI Detection using PPO"""
    def __init__(
        self,
        env,
        model_dir="models",
        log_dir="logs",
        tensorboard_log="logs/tensorboard",
        learning_rate=3e-4
    ):
        # Create directories
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(tensorboard_log, exist_ok=True)

        # PPO hyperparameters - optimized for K-means based environment
        ppo_params = dict(
            policy=ActorCriticPolicy,
            env=env,
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.995,
            ent_coef=1e-3,
            clip_range=0.2,
            vf_coef = 0.5,
            max_grad_norm = 0.5,
            normalize_advantage = True,
            verbose=2,
            tensorboard_log=tensorboard_log,
            policy_kwargs={
                "features_extractor_class": ROIFeatureExtractor,
                "features_extractor_kwargs": dict(features_dim=256),
            }
        )

        # Initialize PPO
        self.model = PPO(**ppo_params)

        self.env = env
        self.model_dir = model_dir
        self.log_dir = log_dir

    def train(self, total_timesteps=1_000_000, callback=None):
        """
        Train the agent
        
        Args:
            total_timesteps: Number of timesteps to train for
            callback: Optional callback(s) to use during training
        """
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def save_model(self, name="roi_kmeans_agent"):
        path = os.path.join(self.model_dir, name)
        self.model.save(path)

    def load_model(self, name="roi_kmeans_agent"):
        path = os.path.join(self.model_dir, name)
        self.model = PPO.load(path, env=self.env)
        
    def predict(self, observation, deterministic=True):
        """Make a prediction with the model"""
        return self.model.predict(observation, deterministic=deterministic)