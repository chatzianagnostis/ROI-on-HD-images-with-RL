"""
ROI Detection Environment Tester - A script to verify the K-means based ROIDetectionEnv works correctly
"""
import os
import cv2
import numpy as np
import time
import argparse
from pathlib import Path

# Import the required classes
from ROIDataset import ROIDataset
from ROIDetectionEnv import ROIDetectionEnv

os.environ['LOKY_MAX_CPU_COUNT'] = '4'

class ROIEnvTester:
    def __init__(self, dataset_path, coco_json_path, crop_size=(640, 640)):
        """
        Initialize the environment tester
        
        Args:
            dataset_path: Path to the dataset directory
            coco_json_path: Path to the COCO annotations
            crop_size: Size of the crop window (width, height)
        """
        # Create output directory
        self.output_dir = Path("env_test_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize the dataset
        print("Loading dataset...")
        self.dataset = ROIDataset(
            dataset_path=dataset_path,
            coco_json_path=coco_json_path,
            image_size=(640, 640),
            annotations_format="coco",
            shuffle=True
        )
        
        # Initialize the environment
        print("Setting up environment with K-means based ROI optimization...")
        self.env = ROIDetectionEnv(
            dataset=self.dataset,
            crop_size=crop_size,
            time_limit=300  # Longer time limit for testing
        )
        
        # Window name for display
        self.window_name = "ROI Environment Test"
        
        # Action history
        self.action_history = []
        self.reward_history = []
        
        print(f"Environment initialized with {len(self.dataset)} samples")
        print(f"Action space: {self.env.action_space}")
        print(f"Observation space: {self.env.observation_space}")
    
    def run_manual_episode(self):
        """
        Run a manual episode where user can control the actions
        """
        print("\nStarting manual episode...")
        print("Controls:")
        print("  W/Up: Move bounding box up")
        print("  S/Down: Move bounding box down")
        print("  A/Left: Move bounding box left")
        print("  D/Right: Move bounding box right")
        print("  Space: Place bounding box")
        print("  R: Remove last bounding box")
        print("  E: End episode")
        print("  Q: Quit")
        print("  K: Show optimal ROIs from K-means (toggle visibility)")
        
        # Reset environment
        obs = self.env.reset()
        done = False
        total_reward = 0
        
        # Toggle for showing optimal ROIs
        show_optimal = True
        
        # Convert dict observation to image for display
        frame = obs['image']
        
        # Draw optimal ROIs if available
        if show_optimal and self.env.optimal_rois:
            for roi in self.env.optimal_rois:
                cv2.rectangle(frame, 
                             (int(roi[0]), int(roi[1])), 
                             (int(roi[0] + roi[2]), int(roi[1] + roi[3])),
                             (0, 0, 255), 1)  # Red color for optimal ROIs
                             
        cv2.imshow(self.window_name, frame)
        
        while not done:
            key = cv2.waitKey(30) & 0xFF
            
            # Map key to action
            action = None
            
            if key == ord('w'):  # W key
                action = 0  # Move up
            elif key == ord('s'):  # S key
                action = 1  # Move down
            elif key == ord('a'):  # A key
                action = 2  # Move left
            elif key == ord('d'):  # D key
                action = 3  # Move right
            elif key == ord(' '):  # Space
                action = 4  # Place bbox
            elif key == ord('r'):  # R
                action = 5  # Remove bbox
            elif key == ord('e'):  # E
                action = 6  # End episode
            elif key == ord('v'):  # V key
                visualization_path = self.env.visualize_reward_landscape("reward_landscape.jpg")
                reward_landscape = cv2.imread(visualization_path)
                cv2.imshow("Reward Landscape", reward_landscape)
            elif key == ord('k'):  # K - toggle optimal ROIs
                show_optimal = not show_optimal
                frame = self.env.render(mode='rgb_array')
                if show_optimal and self.env.optimal_rois:
                    for roi in self.env.optimal_rois:
                        cv2.rectangle(frame, 
                                     (int(roi[0]), int(roi[1])), 
                                     (int(roi[0] + roi[2]), int(roi[1] + roi[3])),
                                     (0, 0, 255), 1)  # Red color for optimal ROIs
                cv2.imshow(self.window_name, frame)
                continue
            elif key == ord('q'):  # Q
                break
                
            if action is not None:
                # Take the action
                action_name = ["Move Up", "Move Down", "Move Left", "Move Right", 
                              "Place Bbox", "Remove Bbox", "End Episode"][action]
                print(f"Action: {action_name}")
                
                # Step environment
                obs, reward, done, info = self.env.step(action)
                
                # Display reward
                print(f"Reward: {reward:.4f}")
                total_reward += reward
                
                # Record action and reward
                self.action_history.append(action)
                self.reward_history.append(reward)
                
                # Render the environment
                frame = self.env.render(mode='rgb_array')
                if frame is not None:
                    # Only draw optimal ROIs if toggled on
                    if show_optimal and self.env.optimal_rois:
                        for roi in self.env.optimal_rois:
                            cv2.rectangle(frame, 
                                         (int(roi[0]), int(roi[1])), 
                                         (int(roi[0] + roi[2]), int(roi[1] + roi[3])),
                                         (0, 0, 255), 1)  # Red color for optimal ROIs
                    cv2.imshow(self.window_name, frame)
                
                # Check if episode ended
                if done:
                    print("\nEpisode finished!")
                    print(f"Total reward: {total_reward:.4f}")
                    if 'metrics' in info:
                        metrics = info['metrics']
                        print("\nFinal Metrics:")
                        
                        # Print metrics for K-means approach
                        if 'coverage_score' in metrics:
                            print(f"  Coverage score: {metrics['coverage_score']:.4f}")
                        if 'roi_matching_score' in metrics:
                            print(f"  ROI matching score: {metrics['roi_matching_score']:.4f}")
                        if 'efficiency_score' in metrics:
                            print(f"  Efficiency score: {metrics['efficiency_score']:.4f}")
                        if 'overlap_penalty' in metrics:
                            print(f"  Overlap penalty: {metrics['overlap_penalty']:.4f}")
                        if 'optimal_count' in metrics and 'placed_count' in metrics:
                            print(f"  Optimal ROI count: {metrics['optimal_count']}")
                            print(f"  Placed ROI count: {metrics['placed_count']}")
                        
                        if 'time' in metrics:
                            time_info = metrics['time']
                            print(f"  Time used: {time_info['elapsed']:.2f}s / {time_info['limit']:.2f}s")
                    
                    # Save final frame
                    if frame is not None:
                        filename = f"final_state_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                        filepath = self.output_dir / filename
                        cv2.imwrite(str(filepath), frame)
                        print(f"Final state saved to {filepath}")
        
        # Close all windows
        cv2.destroyAllWindows()
    
    def run_random_episode(self, num_steps=100, render=True):
        """
        Run an episode with random actions
        
        Args:
            num_steps: Maximum number of steps to run
            render: Whether to render the environment
        """
        print("\nStarting random episode...")
        
        # Reset environment
        obs = self.env.reset()
        done = False
        total_reward = 0
        step = 0
        
        if render:
            frame = self.env.render(mode='rgb_array')
            # Display optimal ROIs from K-means
            if self.env.optimal_rois:
                for roi in self.env.optimal_rois:
                    cv2.rectangle(frame, 
                                 (int(roi[0]), int(roi[1])), 
                                 (int(roi[0] + roi[2]), int(roi[1] + roi[3])),
                                 (0, 0, 255), 1)  # Red color for optimal ROIs
            if frame is not None:
                cv2.imshow(self.window_name, frame)
        
        while not done and step < num_steps:
            # Take random action (with higher probability for movement)
            p = [0.2, 0.2, 0.2, 0.2, 0.1, 0.05, 0.05]  # Probabilities for each action
            action = np.random.choice(7, p=p)
            
            # Step environment
            obs, reward, done, info = self.env.step(action)
            
            # Record action and reward
            self.action_history.append(action)
            self.reward_history.append(reward)
            
            # Update total reward
            total_reward += reward
            step += 1
            
            # Print action and reward
            action_name = ["Move Up", "Move Down", "Move Left", "Move Right", 
                          "Place Bbox", "Remove Bbox", "End Episode"][action]
            print(f"Step {step}: Action: {action_name}, Reward: {reward:.4f}")
            
            # Render if requested
            if render:
                frame = self.env.render(mode='rgb_array')
                # Show optimal ROIs
                if self.env.optimal_rois:
                    for roi in self.env.optimal_rois:
                        cv2.rectangle(frame, 
                                     (int(roi[0]), int(roi[1])), 
                                     (int(roi[0] + roi[2]), int(roi[1] + roi[3])),
                                     (0, 0, 255), 1)  # Red color for optimal ROIs
                if frame is not None:
                    cv2.imshow(self.window_name, frame)
                    cv2.waitKey(1)  # Small delay
        
        print("\nEpisode finished!")
        print(f"Total steps: {step}")
        print(f"Total reward: {total_reward:.4f}")
        
        if 'metrics' in info:
            metrics = info['metrics']
            print("\nFinal Metrics:")
            
            # Print metrics for K-means approach
            if 'coverage_score' in metrics:
                print(f"  Coverage score: {metrics['coverage_score']:.4f}")
            if 'roi_matching_score' in metrics:
                print(f"  ROI matching score: {metrics['roi_matching_score']:.4f}")
            if 'efficiency_score' in metrics:
                print(f"  Efficiency score: {metrics['efficiency_score']:.4f}")
            if 'overlap_penalty' in metrics:
                print(f"  Overlap penalty: {metrics['overlap_penalty']:.4f}")
            if 'optimal_count' in metrics and 'placed_count' in metrics:
                print(f"  Optimal ROI count: {metrics['optimal_count']}")
                print(f"  Placed ROI count: {metrics['placed_count']}")
            
            if 'time' in metrics:
                time_info = metrics['time']
                print(f"  Time used: {time_info['elapsed']:.2f}s / {time_info['limit']:.2f}s")
        
        # Save final frame
        if render and frame is not None:
            filename = f"random_episode_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = self.output_dir / filename
            cv2.imwrite(str(filepath), frame)
            print(f"Final state saved to {filepath}")
        
        # Close all windows
        if render:
            cv2.destroyAllWindows()
    
    def analyze_action_distribution(self):
        """Analyze the distribution of actions and rewards"""
        if not self.action_history:
            print("No actions recorded yet")
            return
        
        print("\nAction Distribution:")
        action_counts = np.bincount(self.action_history, minlength=7)
        action_names = ["Move Up", "Move Down", "Move Left", "Move Right", 
                        "Place Bbox", "Remove Bbox", "End Episode"]
        
        for i, (name, count) in enumerate(zip(action_names, action_counts)):
            print(f"  {name}: {count} ({count/len(self.action_history)*100:.1f}%)")
        
        print("\nReward Statistics:")
        print(f"  Total Reward: {sum(self.reward_history):.4f}")
        print(f"  Average Reward: {np.mean(self.reward_history):.4f}")
        print(f"  Min Reward: {min(self.reward_history):.4f}")
        print(f"  Max Reward: {max(self.reward_history):.4f}")
        
        # Group rewards by action
        rewards_by_action = [[] for _ in range(7)]
        for action, reward in zip(self.action_history, self.reward_history):
            rewards_by_action[action].append(reward)
        
        print("\nAverage Reward by Action:")
        for i, (name, rewards) in enumerate(zip(action_names, rewards_by_action)):
            if rewards:
                print(f"  {name}: {np.mean(rewards):.4f}")
            else:
                print(f"  {name}: N/A")

def main():
    parser = argparse.ArgumentParser(description="ROI Detection Environment Tester (K-means Version)")
    parser.add_argument("--dataset", type=str, required=True,
                      help="Path to the dataset directory")
    parser.add_argument("--annotations", type=str, required=True,
                      help="Path to the COCO JSON annotation file")
    parser.add_argument("--crop_width", type=int, default=640,
                      help="Width of the crop window (default: 640)")
    parser.add_argument("--crop_height", type=int, default=640,
                      help="Height of the crop window (default: 640)")
    parser.add_argument("--mode", type=str, choices=['manual', 'random', 'both'], default='manual',
                      help="Test mode: manual, random, or both (default: manual)")
    
    args = parser.parse_args()
    
    tester = ROIEnvTester(
        dataset_path=args.dataset,
        coco_json_path=args.annotations,
        crop_size=(args.crop_width, args.crop_height)
    )
    
    if args.mode == 'manual' or args.mode == 'both':
        tester.run_manual_episode()
    
    if args.mode == 'random' or args.mode == 'both':
        tester.run_random_episode(num_steps=50)
    
    tester.analyze_action_distribution()

if __name__ == "__main__":
    main()