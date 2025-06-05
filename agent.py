"""
Region of Interest Detection Agent using DINOv2.
Enhanced with action history support.
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
    Enhanced feature extractor for the ROI Detection environment.
    
    This class uses a pre-trained DINOv2-ViT-B/14 with frozen weights to efficiently
    extract features from images, combined with separate networks to process
    bounding box states, action history, position history, and movement patterns.
    """
    
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        """
        Initialize the feature extractor.
        
        Args:
            observation_space: The observation space from the environment
            features_dim: Dimension of the output feature vector
        """
        super(ROIFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # ----- Image Processing Network (DINOv2) -----
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
        bbox_state_dim = np.prod(observation_space.spaces['bbox_state'].shape)  # 400
        current_bbox_dim = np.prod(observation_space.spaces['current_bbox'].shape)  # 4
        bbox_input_dim = bbox_state_dim + current_bbox_dim  # 404

        self.bbox_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(bbox_input_dim, 256),  # 404 -> 256
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),  # Small dropout for regularization
            nn.Linear(256, 128),  # 256 -> 128
            nn.LayerNorm(128), 
            nn.ReLU(),
            nn.Linear(128, 64),  # 128 -> 64
            nn.LayerNorm(64),
            nn.ReLU()
        )
        
        # ----- NEW: Action History Processing -----
        action_history_length = observation_space.spaces['action_history'].shape[0]
        
        # Embedding for actions (8 possible values: 0-6 + padding token 7)
        self.action_embedding = nn.Embedding(8, 16)
        
        # LSTM for action sequence processing
        self.action_lstm = nn.LSTM(
            input_size=16,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
        
        # ----- NEW: Position History Processing -----
        if observation_space.spaces['position_history'].shape[0] > 1:  # If tracking positions
            position_history_dim = np.prod(observation_space.spaces['position_history'].shape)
            self.position_encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(position_history_dim, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Linear(64, 32),  # Reduce to 32 features
                nn.LayerNorm(32),
                nn.ReLU()
            )
            position_features_dim = 32
        else:
            self.position_encoder = None
            position_features_dim = 0
        
        # ----- NEW: Movement Features Processing -----
        movement_features_dim = observation_space.spaces['movement_features'].shape[0]
        self.movement_encoder = nn.Sequential(
            nn.Linear(movement_features_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 16),  # Reduce to 16 features
            nn.LayerNorm(16),
            nn.ReLU()
        )
        
        # ----- Combined Features Network -----
        # Calculate total input dimension
        # DINOv2: 768, bbox: 64, action LSTM: 32, position: 0-32, movement: 16
        total_input_dim = 768 + 64 + 32 + position_features_dim + 16
        
        self.combined_layer = nn.Sequential(
            nn.Linear(total_input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),  # Small dropout for regularization
            nn.Linear(512, features_dim),  # Output to features_dim
            nn.LayerNorm(features_dim),
            nn.ReLU()
        )

        # CRITICAL: Apply custom initialization to prevent saturation
        self._initialize_weights()

    def _initialize_weights(self):
        """Apply custom weight initialization to prevent gradient explosion."""
        
        print("Applying FIXED weight initialization...")
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use smaller initialization for deep layers
                if module.out_features == 512:  # combined_layer.0
                    nn.init.xavier_uniform_(module.weight, gain=0.5)  # Smaller gain
                elif module.out_features == 256:  # combined_layer.4
                    nn.init.xavier_uniform_(module.weight, gain=0.5)
                else:
                    nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))

                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
            
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.0)
                        # Set forget gate bias to 1 (LSTM best practice)
                        with torch.no_grad():  # FIXED: Prevent in-place operation error
                            n = param.size(0)
                            param[n//4:n//2].fill_(1.0)
            
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0, 0.1)
            
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1)
    
    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Preprocess image for DINOv2.
        Applies ImageNet normalization required by DINOv2.
        """
        # Normalize to [0, 1] range if needed
        if image.max() > 1.0:
            image = image.float() / 255.0

        # Apply ImageNet normalization (required for DINOv2)
        image = (image - self.mean) / self.std
        
        return image
        
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = observations['image'].shape[0]
        
        # ===== Process Image =====
        image = observations['image']
        image = self.preprocess_image(image)
        with torch.no_grad():
            image_features = self.dinov2(image)  # (batch_size, 768)
        
        # ===== Process Bbox State =====
        bbox_state = observations['bbox_state'].float()
        current_bbox = observations['current_bbox'].float()
        
        # Combine bbox info
        bbox_combined = torch.cat([
            bbox_state.flatten(start_dim=1),  # Flatten (N, 100, 4) -> (N, 400)
            current_bbox                      # (N, 4)
        ], dim=1)
        
        bbox_features = self.bbox_encoder(bbox_combined)  # (batch_size, 64)
        
        # ===== NEW: Process Action History =====
        action_history = observations['action_history'].long()  # (batch_size, seq_len)
        
        # Embed actions
        action_embeddings = self.action_embedding(action_history)  # (batch_size, seq_len, 16)
        
        # Process with LSTM
        lstm_out, (hidden, cell) = self.action_lstm(action_embeddings)
        action_features = hidden[-1]  # Take last hidden state (batch_size, 32)
        
        # ===== NEW: Process Position History =====
        if self.position_encoder is not None:
            position_history = observations['position_history'].float()
            position_features = self.position_encoder(position_history)  # (batch_size, 32)
        else:
            position_features = None
        
        # ===== NEW: Process Movement Features =====
        movement_features_raw = observations['movement_features'].float()
        movement_features = self.movement_encoder(movement_features_raw)  # (batch_size, 16)

        # # apply scaling
        # image_features = image_features * 0.5  # Reduce DINOv2 dominance
        # bbox_features = bbox_features * 2.0     # Boost bbox importance
        # action_features = action_features * 2.0  # Boost action importance
        # movement_features = movement_features * 3.0  # Really boost movement
        
        # ===== Combine All Features =====
        features_to_combine = [
            image_features,    # (batch_size, 768)
            bbox_features,     # (batch_size, 64)
            action_features,   # (batch_size, 32)
            movement_features  # (batch_size, 16)
        ]
        
        if position_features is not None:
            features_to_combine.append(position_features)  # (batch_size, 32)
        
        combined = torch.cat(features_to_combine, dim=1)
        
        # Final processing
        final_features = self.combined_layer(combined)  # (batch_size, features_dim)
        
        return final_features

def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


class ROIAgent:
    """
    Enhanced Agent for ROI Detection using standard PPO with action history support.
    
    This class handles the training, saving, loading, and inference of a
    reinforcement learning agent for the ROI detection task using standard PPO
    with enhanced observation processing.
    """
    
    def __init__(
        self,
        env,
        model_dir: str = "models",
        log_dir: str = "logs",
        tensorboard_log: Optional[str] = None,
        learning_rate: float = 1e-4
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

        # Network architecture - adjusted for enhanced features
        net_arch = dict(pi=[512, 256, 128], vf=[512, 256, 128]) 
        
        # Configure PPO hyperparameters (optimized for action history)
        ppo_params = {
            # Core components
            "policy": ActorCriticPolicy,
            "env": env,
            "learning_rate": linear_schedule(learning_rate),
            
            # Sample collection parameters
            "n_steps": 512,        
            "batch_size": 16,       # Increased for more stable updates
            
            # Training parameters
            "n_epochs": 10,          
            "gamma": 0.995,         # Standard discount factor
            "gae_lambda": 0.95,     # Standard GAE lambda
            "ent_coef": 0.01,       # Increased exploration for richer observation space
            "clip_range": 0.4,
            "vf_coef": 0.5,         # Standard value function coefficient
            "max_grad_norm": 0.5,   # Gradient clipping
            "normalize_advantage": True ,  # Normalize advantages for stability
            
            # Miscellaneous
            "verbose": 2,
            "tensorboard_log": tensorboard_log,
            # "device": "cpu",
            
            # Custom feature extractor with action history support
            "policy_kwargs": {
                "features_extractor_class": ROIFeatureExtractor,
                "features_extractor_kwargs": {"features_dim": 256},
                "activation_fn": torch.nn.ReLU,    
                "optimizer_class": torch.optim.Adam,  # Standard Adam
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
        """Make a prediction with the model."""
        
        # Get the device of the model
        device = next(self.model.policy.parameters()).device
        
        obs_dict = {}
        
        # Convert and move tensors to correct device
        if 'image' in observation:
            if isinstance(observation['image'], np.ndarray):
                if len(observation['image'].shape) == 3:  # Single image
                    obs_dict['image'] = torch.tensor(observation['image']).unsqueeze(0).permute(0, 3, 1, 2).to(device)
                else:  # Already batched
                    obs_dict['image'] = torch.tensor(observation['image']).permute(0, 3, 1, 2).to(device)
        
        # Convert other components and move to device
        for key in ['bbox_state', 'current_bbox', 'action_history', 'position_history', 'movement_features']:
            if key in observation:
                if isinstance(observation[key], np.ndarray):
                    if len(observation[key].shape) == 1 and key in ['current_bbox', 'action_history', 'movement_features']:
                        obs_dict[key] = torch.tensor(observation[key]).unsqueeze(0).to(device)
                    elif len(observation[key].shape) == 2 and key in ['bbox_state', 'position_history']:
                        obs_dict[key] = torch.tensor(observation[key]).unsqueeze(0).to(device)
                    else:  # Already batched
                        obs_dict[key] = torch.tensor(observation[key]).to(device)
        
        action, _states = self.model.predict(obs_dict, deterministic=deterministic)
        return action