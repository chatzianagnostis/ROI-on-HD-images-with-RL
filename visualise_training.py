"""
Episode Visualization Script for ROI Detection Training

This script takes a training run ID and creates a video showing all episodes
from that run, replaying the actions step by step with overlaid information.
"""

import os
import json
import glob
import cv2
import numpy as np
import argparse
from pathlib import Path
import time
from typing import List, Dict, Any

from ROIDataset import ROIDataset
from ROIDetectionEnv import ROIDetectionEnv


class EpisodeVisualizer:
    """
    Visualizes training episodes by replaying actions and creating video frames.
    """
    
    def __init__(self, dataset_path: str, coco_json_path: str, 
                 crop_size: tuple = (640, 640), time_limit: int = 60):
        """
        Initialize the episode visualizer.
        
        Args:
            dataset_path: Path to the dataset directory
            coco_json_path: Path to the COCO annotations
            crop_size: Size of the crop window
            time_limit: Time limit for episodes
        """
        # Action names for display
        self.action_names = [
            "Move Up", "Move Down", "Move Left", "Move Right",
            "Place Bbox", "Remove Bbox", "End Episode"
        ]
        
        # Initialize dataset with shuffle=False for consistent ordering
        print("Loading dataset...")
        self.dataset = ROIDataset(
            dataset_path=dataset_path,
            coco_json_path=coco_json_path,
            image_size=(640, 640),
            annotations_format="coco",
            shuffle=False  # Important: consistent ordering
        )
        
        # Initialize environment
        print("Setting up environment...")
        self.env = ROIDetectionEnv(
            dataset=self.dataset,
            crop_size=crop_size,
            time_limit=time_limit
        )
        
        print(f"Environment ready with {len(self.dataset)} samples")
    
    def load_episode_data(self, ppo_dir: str) -> List[Dict[str, Any]]:
        """
        Load all episode JSON files from the ppo_ directory.
        
        Args:
            ppo_dir: Path to the ppo_ directory containing episode JSON files
            
        Returns:
            List of episode data dictionaries, sorted by episode number
        """
        episode_files = sorted(glob.glob(os.path.join(ppo_dir, "episode_*.json")))
        if not episode_files:
            raise ValueError(f"No episode data found in {ppo_dir}")
        
        episodes = []
        for ep_file in episode_files:
            try:
                with open(ep_file, 'r') as f:
                    episode_data = json.load(f)
                    episodes.append(episode_data)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {ep_file}")
            except Exception as e:
                print(f"Warning: Error processing {ep_file}: {e}")
        
        # Sort by episode number
        episodes.sort(key=lambda x: x.get("episode", 0))
        print(f"Loaded {len(episodes)} episodes")
        return episodes
    
    def create_info_overlay(self, frame: np.ndarray, episode_data: Dict[str, Any], 
                           step_idx: int, action: int, reward: float) -> np.ndarray:
        """
        Add text overlay with episode information to the frame.
        
        Args:
            frame: The base frame to overlay text on
            episode_data: Episode data dictionary
            step_idx: Current step index in the episode
            action: Action taken at this step
            reward: Reward received for this action
            
        Returns:
            Frame with text overlay
        """
        overlay_frame = frame.copy()
        
        # Define text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Background color for text (semi-transparent black)
        overlay = overlay_frame.copy()
        
        # Episode info (top-left)
        episode_num = episode_data.get("episode", "Unknown")
        total_reward = episode_data.get("total_reward", 0)
        step_count = episode_data.get("step_count", 0)
        
        info_lines = [
            f"Episode: {episode_num}",
            f"Step: {step_idx + 1}/{step_count}",
            f"Total Reward: {total_reward:.2f}",
            f"Action: {self.action_names[action]}",
            f"Step Reward: {reward:.4f}"
        ]
        
        # Draw background rectangle for text
        text_height = 25
        rect_height = len(info_lines) * text_height + 20
        cv2.rectangle(overlay, (10, 10), (400, rect_height), (0, 0, 0), -1)
        
        # Draw text lines
        for i, line in enumerate(info_lines):
            y_pos = 35 + i * text_height
            cv2.putText(overlay, line, (20, y_pos), font, font_scale, (255, 255, 255), thickness)
        
        # Metrics info (top-right)
        if "metrics" in episode_data:
            metrics = episode_data["metrics"]
            metrics_lines = []
            
            if "coverage_score" in metrics:
                metrics_lines.append(f"Coverage: {metrics['coverage_score']:.3f}")
            if "roi_matching_score" in metrics:
                metrics_lines.append(f"ROI Match: {metrics['roi_matching_score']:.3f}")
            if "optimal_count" in metrics and "placed_count" in metrics:
                metrics_lines.append(f"ROIs: {metrics['placed_count']}/{metrics['optimal_count']}")
            
            if metrics_lines:
                # Draw background rectangle for metrics
                metrics_rect_width = 250
                metrics_rect_height = len(metrics_lines) * text_height + 20
                frame_width = overlay_frame.shape[1]
                cv2.rectangle(overlay, 
                            (frame_width - metrics_rect_width - 10, 10), 
                            (frame_width - 10, metrics_rect_height), 
                            (0, 0, 0), -1)
                
                # Draw metrics text
                for i, line in enumerate(metrics_lines):
                    y_pos = 35 + i * text_height
                    x_pos = frame_width - metrics_rect_width + 10
                    cv2.putText(overlay, line, (x_pos, y_pos), font, font_scale, (255, 255, 255), thickness)
        
        # Blend the overlay with the original frame
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, overlay_frame, 1 - alpha, 0, overlay_frame)
        
        return overlay_frame
    
    def replay_episode(self, episode_data: Dict[str, Any]) -> List[np.ndarray]:
        """
        Replay a single episode and return frames.
        
        Args:
            episode_data: Episode data dictionary
            
        Returns:
            List of frames from the episode replay
        """
        actions = episode_data.get("actions", [])
        rewards = episode_data.get("rewards", [])
        
        if len(actions) != len(rewards):
            print(f"Warning: Action/reward mismatch in episode {episode_data.get('episode', 'Unknown')}")
            return []
        
        frames = []
        
        try:
            # Reset environment to start episode
            obs = self.env.reset()
            
            # Get initial frame
            initial_frame = self.env.render(mode='rgb_array')
            if initial_frame is not None:
                # Add overlay showing episode start
                overlay_frame = self.create_info_overlay(
                    initial_frame, episode_data, -1, -1, 0.0
                )
                frames.append(overlay_frame)
            
            # Replay each action
            for step_idx, (action, reward) in enumerate(zip(actions, rewards)):
                # Take the action
                obs, step_reward, done, info = self.env.step(action)
                
                # Render the state after action
                frame = self.env.render(mode='rgb_array')
                if frame is not None:
                    # Add informational overlay
                    overlay_frame = self.create_info_overlay(
                        frame, episode_data, step_idx, action, reward
                    )
                    frames.append(overlay_frame)
                
                # If episode is done, break
                if done:
                    break
                    
                # Add a small delay between frames for better visualization
                time.sleep(0.05)
                
        except Exception as e:
            print(f"Error replaying episode {episode_data.get('episode', 'Unknown')}: {e}")
            return frames
        
        return frames
    
    def create_video(self, episodes: List[Dict[str, Any]], output_path: str, 
                    fps: int = 10) -> None:
        """
        Create a video from all episodes.
        
        Args:
            episodes: List of episode data dictionaries
            output_path: Path where to save the output video
            fps: Frames per second for the output video
        """
        if not episodes:
            print("No episodes to visualize")
            return
        
        print(f"Creating video with {len(episodes)} episodes...")
        
        # Initialize video writer (we'll set it up after getting the first frame)
        video_writer = None
        total_frames = 0
        
        try:
            for ep_idx, episode_data in enumerate(episodes):
                print(f"Processing episode {ep_idx + 1}/{len(episodes)}: "
                      f"Episode {episode_data.get('episode', 'Unknown')}")
                
                # Replay episode and get frames
                frames = self.replay_episode(episode_data)
                
                if not frames:
                    print(f"No frames generated for episode {episode_data.get('episode', 'Unknown')}")
                    continue
                
                # Initialize video writer with first frame
                if video_writer is None:
                    frame_height, frame_width = frames[0].shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(
                        output_path, fourcc, fps, (frame_width, frame_height)
                    )
                    print(f"Video writer initialized: {frame_width}x{frame_height} at {fps} FPS")
                
                # Write frames to video
                for frame in frames:
                    # Convert RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    video_writer.write(frame_bgr)
                    total_frames += 1
                
                # Add separator frames between episodes (black frames with episode info)
                if ep_idx < len(episodes) - 1:  # Don't add separator after last episode
                    separator_frame = np.zeros_like(frames[0])
                    
                    # Add text to separator frame
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text = f"End of Episode {episode_data.get('episode', 'Unknown')} - " \
                           f"Total Reward: {episode_data.get('total_reward', 0):.2f}"
                    
                    # Calculate text size and position
                    text_size = cv2.getTextSize(text, font, 1, 2)[0]
                    text_x = (separator_frame.shape[1] - text_size[0]) // 2
                    text_y = (separator_frame.shape[0] + text_size[1]) // 2
                    
                    cv2.putText(separator_frame, text, (text_x, text_y), 
                               font, 1, (255, 255, 255), 2)
                    
                    # Add separator frames (hold for 1 second)
                    separator_bgr = cv2.cvtColor(separator_frame, cv2.COLOR_RGB2BGR)
                    for _ in range(fps):  # 1 second worth of frames
                        video_writer.write(separator_bgr)
                        total_frames += 1
        
        except Exception as e:
            print(f"Error creating video: {e}")
        
        finally:
            if video_writer is not None:
                video_writer.release()
                print(f"Video saved to {output_path}")
                print(f"Total frames: {total_frames}")
                print(f"Video duration: {total_frames / fps:.2f} seconds")
            else:
                print("Failed to create video - no frames generated")
    
    def visualize_run(self, log_dir: str, run_id: str, output_dir: str = "episode_videos") -> None:
        """
        Visualize all episodes from a specific training run.
        
        Args:
            log_dir: Base log directory
            run_id: Run ID to visualize
            output_dir: Output directory for videos
        """
        # Find the run directory
        runs_dir = os.path.join(log_dir, "runs")
        if not os.path.exists(runs_dir):
            raise ValueError(f"Runs directory not found: {runs_dir}")
        
        # Find matching run directories
        run_dirs = [d for d in os.listdir(runs_dir) 
                   if os.path.isdir(os.path.join(runs_dir, d)) and run_id in d]
        
        if not run_dirs:
            raise ValueError(f"No run directories found matching '{run_id}' in {runs_dir}")
        
        # Use the first matching run (or most recent if multiple matches)
        selected_run = sorted(run_dirs)[-1]
        run_path = os.path.join(runs_dir, selected_run)
        ppo_path = os.path.join(run_path, "ppo_")
        
        print(f"Visualizing run: {selected_run}")
        print(f"Loading episode data from: {ppo_path}")
        
        # Load episode data
        episodes = self.load_episode_data(ppo_path)
        
        if not episodes:
            raise ValueError("No episodes found to visualize")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"episodes_visualization_{selected_run}_{timestamp}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        # Create the video
        self.create_video(episodes, output_path)
        
        return output_path


def main():
    """Main function to run episode visualization."""
    parser = argparse.ArgumentParser(description="Visualize training episodes from a specific run")
    parser.add_argument("--run_id", required=True, 
                       help="Run ID (substring) to visualize")
    parser.add_argument("--log_dir", default="logs", 
                       help="Base log directory (default: logs)")
    parser.add_argument("--dataset", required=True,
                       help="Path to the dataset directory")
    parser.add_argument("--annotations", required=True,
                       help="Path to the COCO JSON annotations")
    parser.add_argument("--output_dir", default="episode_videos",
                       help="Output directory for videos (default: episode_videos)")
    parser.add_argument("--fps", type=int, default=5,
                       help="Video framerate (default: 5)")
    parser.add_argument("--crop_width", type=int, default=640,
                       help="Crop width (default: 640)")
    parser.add_argument("--crop_height", type=int, default=640,
                       help="Crop height (default: 640)")
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = EpisodeVisualizer(
        dataset_path=args.dataset,
        coco_json_path=args.annotations,
        crop_size=(args.crop_width, args.crop_height)
    )
    
    try:
        # Visualize the run
        output_path = visualizer.visualize_run(
            log_dir=args.log_dir,
            run_id=args.run_id,
            output_dir=args.output_dir
        )
        
        print(f"\n{'='*60}")
        print("VISUALIZATION COMPLETE")
        print(f"{'='*60}")
        print(f"Output video: {output_path}")
        print(f"You can now view the episode replay video.")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())