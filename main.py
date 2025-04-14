"""
ROI Detection Training Script
"""
import os
import time
import torch
from stable_baselines3.common.callbacks import CheckpointCallback

# Import required classes
from ROIDataset import ROIDataset
from ROIDetectionEnv import ROIDetectionEnv
from agent import ROIAgent

def main():
    print("ROI Detection Agent - Training")
    print("=============================")
    
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs("models", exist_ok=True)
    
    # 1. Setup dataset and environment
    print("Setting up environment...")
    dataset_path = "G:\\rl\\dhd_campus\\dhd_campus_train_images_part1\\dhd_campus\\images\\train"
    coco_json_path = "G:\\rl\\dhd_campus\\dhd_pedestrian_campus_trainval_annos\\dhd_pedestrian\\ped_campus\\annotations\\dhd_pedestrian_campus_train.json"
    
    # Create dataset with 640x640 images as requested
    dataset = ROIDataset(
        dataset_path=dataset_path,
        coco_json_path=coco_json_path,
        image_size=(640, 640),
        annotations_format="coco",
        shuffle=True
    )
    
    # Create environment
    env = ROIDetectionEnv(
        dataset=dataset,
        crop_size=(640, 640),
        yolo_model_path="yolov8n.pt"
    )
    
    print(f"Environment ready - dataset has {len(dataset)} samples")
    
    # 2. Create and train agent
    agent = ROIAgent(
        env=env,
        model_dir="models",
        log_dir="logs"
    )
    
    # Create a CheckpointCallback to save every 10,000 timesteps
    checkpoint_callback = CheckpointCallback(
        save_freq=15000,  # Save every 10,000 steps
        save_path="models/checkpoints/",
        name_prefix="roi_model",
        verbose=2
    )
    
    # Train and save model
    print("\nTraining agent...")
    start_time = time.time()
    
    # You can adjust this number based on your computational resources
    agent.train(total_timesteps=1000000, callback=checkpoint_callback)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save final model
    agent.save_model("final_model")
    print("Final model saved to models/final_model.zip")

if __name__ == "__main__":
    main()