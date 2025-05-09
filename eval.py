import time
import os
import cv2
from ROIDataset import ROIDataset
from ROIDetectionEnv import ROIDetectionEnv
from agent import ROIAgent

os.environ['LOKY_MAX_CPU_COUNT'] = '4'

# Setup dataset and environment
dataset = ROIDataset(
    # dataset_path="G:\\rl\\dhd_campus\\dhd_campus_train_images_part1\\dhd_campus\\images\\train",
    # coco_json_path="G:\\rl\\dhd_campus\\dhd_pedestrian_campus_trainval_annos\\dhd_pedestrian\\ped_campus\\annotations\\dhd_pedestrian_campus_train.json",
    dataset_path = "G:\\rl\\overfit\\images",
    coco_json_path = "G:\\rl\\overfit\\overfit.json",
    image_size=(640, 640),
    annotations_format="coco",
    shuffle=True
)

env = ROIDetectionEnv(
    dataset=dataset,
    crop_size=(640, 640),
    time_limit=60
)

# Load trained agent
agent = ROIAgent(env)
agent.load_model("final_roi_model")

# Create output directory for videos
output_dir = "evaluation_videos"
os.makedirs(output_dir, exist_ok=True)

# Evaluate
num_episodes = 10
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    total_reward = 0
    
    print(f"Episode {episode+1}")
    
    # Setup video writer
    video_path = os.path.join(output_dir, f"episode_{episode+1}.mp4")
    image_shape = env.render(mode='rgb_array').shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    fps = 10  # Frames per second
    video_writer = cv2.VideoWriter(
        video_path, 
        fourcc, 
        fps, 
        (image_shape[1], image_shape[0])  # Width, Height
    )
    
    step_count = 0
    while not done:
        # Get frame for video
        frame = env.render(mode='rgb_array')
        video_writer.write(frame)
        
        # Display frame (optional - comment out if not needed)
        cv2.imshow('ROI Detection Evaluation', frame)
        cv2.waitKey(1)
        
        # Agent step
        action, _ = agent.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        # Add a slight delay to limit the frame rate for viewing
        time.sleep(0.1)
        
        step_count += 1
        if step_count % 10 == 0:
            print(f"  Step {step_count}, Reward: {reward:.4f}")
    
    # Add final frame to video
    final_frame = env.render(mode='rgb_array')
    video_writer.write(final_frame)
    
    # Release video writer
    video_writer.release()
    
    print(f"Episode {episode+1} completed. Total reward: {total_reward:.4f}")
    print(f"Video saved to {video_path}")
    
    if 'metrics' in info:
        coverage = info['metrics'].get('coverage_score', 0) * 100
        roi_matching = info['metrics'].get('roi_matching_score', 0) * 100
        print(f"  Coverage: {coverage:.1f}%, ROI matching: {roi_matching:.1f}%")
        print(f"  Optimal ROIs: {info['metrics'].get('optimal_count', 0)}, Placed ROIs: {info['metrics'].get('placed_count', 0)}")

# Close any open windows
cv2.destroyAllWindows()
print(f"Evaluation complete. Videos saved to {output_dir}")