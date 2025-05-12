import os
import time
import torch
from stable_baselines3.common.callbacks import CheckpointCallback
# Assuming these custom utils are available and working
# from utils.callbacks import DynamicRewardShapingCallback 
# from utils.loggers import TrainingLogger

# Import required classes
from ROIDataset import ROIDataset # Assuming this file exists and is correct
from ROIDetectionEnv import ROIDetectionEnv
from agent import ROIAgent

os.environ['LOKY_MAX_CPU_COUNT'] = '4' # Setting for joblib, used by SB3 for parallel envs if applicable

def main():
    print("ROI Detection Agent - Training with MaskablePPO") # Updated print
    print("==============================================")
    
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # 1. Setup dataset and environment
    print("Setting up environment...")
    # Ensure these paths are correct for your system
    dataset_path="G:\\rl\\dhd_campus\\dhd_campus_train_images_part1\\dhd_campus\\images\\train"
    coco_json_path="G:\\rl\\dhd_campus\\dhd_pedestrian_campus_trainval_annos\\dhd_pedestrian\\ped_campus\\annotations\\dhd_pedestrian_campus_train.json"

    # Example for a smaller dataset for quick testing (overfitting)
    # dataset_path = "G:\\rl\\overfit\\images"
    # coco_json_path = "G:\\rl\\overfit\\overfit.json"
    
    # Create dataset with 640x640 images
    dataset = ROIDataset(
        dataset_path=dataset_path,
        coco_json_path=coco_json_path,
        image_size=(640, 640), # Image size for resizing
        annotations_format="coco",
        shuffle=True
    )
    
    # Create environment
    env = ROIDetectionEnv(
        dataset=dataset,
        crop_size=(640, 640), # Crop size for ROIs (should relate to model if patches are extracted)
        time_limit=60  # 1 minute per episode (adjust as needed)
    )
    
    print(f"Environment ready - dataset has {len(dataset)} samples") # Uses len(dataset)
    
    # 2. Create and train agent
    log_dir = "logs"  # Single log directory
    agent = ROIAgent(
        env=env,
        model_dir="models",
        log_dir=log_dir,
        tensorboard_log=log_dir # Use the same directory for TensorBoard
    )
    
    # Create a CheckpointCallback to save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=20000, # Save every 20000 steps
        save_path="models/checkpoints/",
        name_prefix="roi_maskable_model", # Updated prefix
        verbose=2
    )
   
    # Dynamic reward shaping callback (if you are using it)
    from utils.callbacks import DynamicRewardShapingCallback # Make sure this import is active if using
    shaping_callback = DynamicRewardShapingCallback(
        check_freq=25000,
        window_size=100,
        initial_coeff=0.1, 
        boost_coeff=0.5, 
        verbose=1
    )
    
    # Training logger callback (if you are using it)
    from utils.loggers import TrainingLogger # Make sure this import is active if using
    logger_callback = TrainingLogger(log_dir=log_dir, verbose=1)

    # Combine callbacks
    callbacks = [checkpoint_callback, shaping_callback, logger_callback]
    # callbacks = [checkpoint_callback] # Using only checkpoint callback for simplicity here
    # If you have the custom callbacks, ensure they are compatible and add them back.

    # Train and save model
    print("\nTraining agent...")
    start_time_train = time.time() # Renamed for clarity
    
    agent.train(total_timesteps=10_000_000, callback=callbacks) # Example: 10M timesteps
    
    training_duration = time.time() - start_time_train # Renamed
    print(f"Training completed in {training_duration:.2f} seconds ({training_duration/3600:.2f} hours)")
    
    # Save final model
    agent.save_model("final_roi_maskable_model") # Updated name
    print("Final model saved to models/final_roi_maskable_model.zip")

if __name__ == "__main__":
    main()