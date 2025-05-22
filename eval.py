import time
import os
import cv2
import numpy as np
import torch
from ROIDataset import ROIDataset
from ROIDetectionEnv import ROIDetectionEnv
from agent import ROIAgent

# Setup dataset
dataset = ROIDataset(
    dataset_path="G:\\rl\\overfit\\images",
    coco_json_path="G:\\rl\\overfit\\overfit.json",
    image_size=(640, 640),
    annotations_format="coco",
    shuffle=True
)

# Create the basic environment
env = ROIDetectionEnv(
    dataset=dataset,
    crop_size=(640, 640),
    time_limit=60
)

# Load trained agent
agent = ROIAgent(env)
agent.load_model("checkpoints/roi_ppo_model_60000_steps")  # Updated model name

# Create output directory for videos
output_dir = "evaluation_videos"
os.makedirs(output_dir, exist_ok=True)

# Evaluate
num_episodes = 10
for episode in range(num_episodes):
    # Reset environment
    obs = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    current_info = {}
    
    print(f"Episode {episode+1}")
    
    # Get initial frame to set up video writer
    initial_frame = env.render(mode='rgb_array')
    if initial_frame is None:
        print(f"Error: Failed to render initial frame for episode {episode+1}")
        continue
    
    # Set up video writer
    video_path = os.path.join(output_dir, f"episode_{episode+1}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10
    video_writer = cv2.VideoWriter(
        video_path,
        fourcc,
        fps,
        (initial_frame.shape[1], initial_frame.shape[0])
    )
    
    # Main episode loop
    while not done:
        # Render current state
        frame = env.render(mode='rgb_array')
        if frame is None:
            print(f"Warning: Failed to render frame at step {step_count}")
            continue
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)
        
        # Display frame (optional)
        cv2.imshow('ROI Detection Evaluation', frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            done = True
            print("Evaluation aborted by user")
            break
        
        # Convert observation to the format expected by the model
        # (batched tensor with correct dimensions)
        obs_dict = {
            'image': torch.tensor(obs['image']).unsqueeze(0).permute(0, 3, 1, 2),
            'bbox_state': torch.tensor(obs['bbox_state']).unsqueeze(0)
        }
        
        # Standard prediction (no action masks)
        action, _ = agent.model.predict(obs_dict, deterministic=True)
        
        # Ensure action is a scalar if the agent returns a batch of 1
        if isinstance(action, np.ndarray) and action.shape == (1,):
            action = action.item()
        
        # Take a step in the environment
        obs, reward, done, info = env.step(action)
        
        total_reward += reward
        current_info = info  # Keep track of the latest info
        step_count += 1
        
        # Print step information
        print(f"  Step {step_count}, Action: {action}, Reward: {reward:.4f}")
        
        # Short delay to make visualization easier to follow
        time.sleep(0.05)
    
    # Clean up
    video_writer.release()
    
    print(f"Episode {episode+1} completed. Total reward: {total_reward:.4f}")
    print(f"Video saved to {video_path}")
    
    # Print metrics if available
    if 'metrics' in current_info:
        coverage = current_info['metrics'].get('coverage_score', 0) * 100
        roi_matching = current_info['metrics'].get('roi_matching_score', 0) * 100
        print(f"  Coverage: {coverage:.1f}%, ROI matching: {roi_matching:.1f}%")
        print(f"  Optimal ROIs: {current_info['metrics'].get('optimal_count', 0)}, Placed ROIs: {current_info['metrics'].get('placed_count', 0)}")

# Close all windows
cv2.destroyAllWindows()
print(f"Evaluation complete. Videos saved to {output_dir}")