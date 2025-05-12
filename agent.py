import os
import numpy as np
import torch
import torch.nn as nn
from sb3_contrib import MaskablePPO # Using MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy # Policy for MaskablePPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision import models
from gym import spaces
from typing import Dict, List, Optional # Not explicitly used by this agent class

class ROIFeatureExtractor(BaseFeaturesExtractor):
    """Feature extractor for the ROI Detection environment using ResNet18"""
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super(ROIFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # Load ResNet18 and remove final FC layer
        self.cnn = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1]) # Remove the final fully connected layer
        
        # Freeze most layers to speed up training (e.g., all but the last few conv layers/blocks)
        # The exact number of layers to freeze might need tuning.
        # Here, assuming parameters are in order and we unfreeze roughly the last block (e.g. layer4).
        # Counting children of nn.Sequential:
        # 0: conv1, 1: bn1, 2: relu, 3: maxpool, 4: layer1, 5: layer2, 6: layer3, 7: layer4, 8: avgpool (removed with [:-1])
        # If cnn is now Sequential(*list(orig_resnet.children())[:-1]), then children are:
        # [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool] (if avgpool was kept, but it's usually removed by [:-1] above or by taking children()[:-2] on original ResNet)
        # If self.cnn = nn.Sequential(*list(self.cnn.children())[:-1]) refers to the original ResNet's children minus FC, then avgpool is the last.
        # Let's assume the intention is to unfreeze the last convolutional block (layer4) and the preceding layers are frozen.
        # ResNet18 structure: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc
        # After `*list(self.cnn.children())[:-1]`: removes fc. Then `*list(self.cnn.children())[:-1]` again removes avgpool.
        # So self.cnn contains [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4]
        # `list(self.cnn.parameters())[:-4]` might not be robust. A better way is to iterate through named modules.
        # For simplicity, keeping the original logic, but this might need review for optimal fine-tuning.
        num_params_to_unfreeze = 10 # Example: unfreeze parameters of last few layers.
                                   # This depends on how parameters are grouped.
        # A more robust way to freeze:
        # for name, param in self.cnn.named_parameters():
        #    if not name.startswith("7."): # Assuming layer4 is child '7' in the nn.Sequential
        #        param.requires_grad = False
        # Given the original `[:-4]`, let's stick to it for now.
        for param in list(self.cnn.parameters())[:-num_params_to_unfreeze]: # Adjust number if needed
            param.requires_grad = False
            
        # Network for bbox state processing
        # Ensure bbox_shape correctly reflects the flattened dimension
        bbox_input_dim = np.prod(observation_space.spaces['bbox_state'].shape)
        self.bbox_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(bbox_input_dim, 64), # Use calculated input dimension
            nn.ReLU()
        )
        
        # Combine features from CNN (512 for ResNet18) and bbox_encoder (64)
        self.combined_layer = nn.Sequential(
            nn.Linear(512 + 64, features_dim), # 512 from ResNet18's output before FC
            nn.ReLU()
        )
        
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Get the image from observations
        image = observations['image'] # Expected shape (N, H, W, C) or (N, C, H, W)
        
        # SB3 passes images as (N, H, W, C) for Box space, PyTorch CNNs expect (N, C, H, W)
        # Also, ensure normalization matches what ResNet18 expects
        # image = image.permute(0, 3, 1, 2) # Change to (N, C, H, W)
        image = image.float() / 255.0 # Normalize to [0, 1]
        # Further normalization (mean, std) as used during ResNet training can be beneficial
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # image = normalize(image) # This would require torchvision.transforms

        image_features = self.cnn(image)
        image_features = torch.flatten(image_features, start_dim=1)
        
        # Process bbox state
        bbox_state = observations['bbox_state'].float()
        # The Flatten layer in bbox_encoder handles unbatched/batched cases.
        # No need for manual reshape if Flatten() is the first layer.
        # if len(bbox_state.shape) > 2: # This was for manual flattening logic
        #     bbox_state = bbox_state.reshape(bbox_state.shape[0], -1)
        # else:
        #     bbox_state = bbox_state.reshape(1, -1) # This would fail if batch_size > 1 and input is already (N, Flat)
            
        bbox_features = self.bbox_encoder(bbox_state) # Flatten is handled by nn.Flatten()
        
        # Combine features from both inputs
        combined = torch.cat([image_features, bbox_features], dim=1)
        return self.combined_layer(combined)

class ROIAgent:
    """Agent for ROI Detection using MaskablePPO"""
    def __init__(
        self,
        env,
        model_dir="models",
        log_dir="logs",
        tensorboard_log=None, 
        learning_rate=3e-4
    ):
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        if tensorboard_log is None:
            tensorboard_log = log_dir # Default tensorboard_log to log_dir
        else:
            os.makedirs(tensorboard_log, exist_ok=True)

        # MaskablePPO hyperparameters
        ppo_params = dict(
            policy=MaskableActorCriticPolicy,  # Use the maskable policy
            env=env,
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=0.005,
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

        # Initialize MaskablePPO
        self.model = MaskablePPO(**ppo_params)

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

    def save_model(self, name="roi_maskable_agent"): # Updated default name
        path = os.path.join(self.model_dir, name)
        self.model.save(path)
        print(f"Model saved to {path}.zip")


    def load_model(self, name="roi_maskable_agent"): # Updated default name
        path = os.path.join(self.model_dir, name)
        # Load with MaskablePPO
        self.model = MaskablePPO.load(path, env=self.env)
        print(f"Model loaded from {path}.zip")

        
    def predict(self, observation, deterministic=True):
        """Make a prediction with the model, applying action masks."""
        action_masks_pred = None # Renamed to avoid conflict

        # Get action masks from the environment for prediction
        # This assumes self.env is the base environment or a compatible wrapper.
        # If self.env is a VecEnv, MaskablePPO handles mask collection during training.
        # For single observation predict, we might need to fetch it manually if env is not directly the base env.
        # For simplicity, assuming self.env is the direct ROIDetectionEnv instance.
        # Note: When using model directly (e.g. after loading), env passed to load() is used for action space etc.
        # but for action_masks in predict, we need the current mask from the actual env instance.
        
        # If the environment instance used for predict is the same as self.env:
        if hasattr(self.env, 'action_masks') and callable(self.env.action_masks):
            action_masks_pred = self.env.action_masks() 
        # else:
            # This branch might be needed if self.env is a VecEnv and we are predicting on an unwrapped observation.
            # For training, MaskablePPO handles VecEnvs correctly.
            # print("Warning: Could not get action_masks for predict. Prediction might not be masked.")

        action, _states = self.model.predict(
            observation,
            deterministic=deterministic,
            action_masks=action_masks_pred # Pass the fetched masks
        )
        return action # Typically return just the action