# main.py - Detailed Documentation

## Overview
`main.py` serves as the entry point for the ROI detection training system. It orchestrates the loading of datasets, environment setup, agent initialization, and the training process.

## Main Function

```python
def main():
```

- **Lines 11-13**: Print program header
  - Indicates the program's purpose (ROI Detection Agent Training)

- **Lines 16-18**: Check and report CUDA availability
  - Determines whether training will use GPU or CPU
  - Prints the selected device

- **Lines 21-22**: Create output directories
  - Ensures "models" directory exists for saving trained models

- **Lines 25-36**: Setup dataset and environment
  - **Lines 26-27**: Print setup information
  - **Lines 28-29**: Define paths to dataset and annotations
    - Image directory path
    - COCO JSON annotation path
  - **Lines 32-37**: Create ROIDataset instance
    - Configures image size to 640x640
    - Specifies COCO annotation format
    - Enables dataset shuffling

- **Lines 40-44**: Create environment
  - Creates an ROIDetectionEnv instance
  - Configures crop size to match dataset
  - Sets episode time limit to 120 seconds (2 minutes)

- **Line 46**: Print environment readiness and dataset size

- **Lines 49-53**: Create and configure the agent
  - Initializes the ROIAgent with the environment
  - Specifies directories for models and logs

- **Lines 56-61**: Create a checkpoint callback
  - Configures model saving every 15,000 timesteps
  - Sets save path and name prefix
  - Enables verbose logging

- **Lines 64-70**: Train and time the agent
  - **Lines 64-65**: Print training start message
  - **Line 66**: Record start time
  - **Line 69**: Call agent's train method
    - Sets total timesteps to 1,000,000
    - Passes the checkpoint callback
  - **Line 70**: Calculate and print training time

- **Lines 73-74**: Save the final model
  - Saves with name "final_kmeans_model"
  - Prints confirmation message

## Entry Point

```python
if __name__ == "__main__":
    main()
```

- **Lines 76-77**: Standard Python entry point
  - Ensures main() is only called when the script is run directly

## Configuration Details

### Training Configuration
- **Total timesteps**: 1,000,000
- **Checkpoint frequency**: Every 15,000 timesteps
- **Checkpoint path**: ./models/checkpoints/
- **Final model name**: final_kmeans_model

## Training Workflow

The main function implements a clear training workflow:

1. **Hardware Detection**: Identifies available computing resources (GPU/CPU)
2. **Directory Setup**: Creates necessary output directories
3. **Dataset Initialization**: Loads the dataset with appropriate preprocessing
4. **Environment Creation**: Sets up the reinforcement learning environment
5. **Agent Configuration**: Initializes the agent with appropriate parameters
6. **Callback Setup**: Configures periodic checkpointing
7. **Training Execution**: Runs the actual training process
8. **Model Persistence**: Saves the final trained model

## Methodology

The training process follows these methodological principles:

1. **Resource Optimization**: Adapts to available hardware (CUDA vs CPU)
2. **Progress Tracking**: Uses checkpoint callbacks to save intermediate models
3. **Time Management**: Records and reports training duration
4. **Organized Output**: Maintains clear directory structure for artifacts
5. **Standardized Configuration**: Uses fixed settings for reproducible results

The approach prioritizes training stability and reproducibility while providing adequate monitoring through verbose logging and checkpoints.

## Usage

The script is designed to be run directly as the main entry point:

```bash
python main.py
```

The script assumes dataset paths are correctly configured and dependencies are installed. It produces a trained model that can later be used for inference or further fine-tuning. Dataset Configuration
- **Image size**: 640x640 pixels
- **Annotations format**: COCO JSON
- **Dataset shuffling**: Enabled

### Environment Configuration
- **Crop size**: 640x640 pixels
- **Episode time limit**: 120 seconds

### Agent Configuration
- Paths configured for organized output:
  - **Models**: ./models/
  - **Logs**: ./logs/

###