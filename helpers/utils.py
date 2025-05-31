import cv2
import os
import numpy as np
from typing import List, Optional, Tuple
import os

from pathlib import Path
from . import FrameSynthesizer, RAFTSynthesizer

def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    os.makedirs(output_folder, exist_ok=True)
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{output_folder}/frame_{idx:04d}.png", frame)
        idx += 1
    cap.release()


def save_video_from_frames(folder, output_path, fps=30):
    images = sorted([img for img in os.listdir(folder) if img.endswith('.png')])
    frame = cv2.imread(os.path.join(folder, images[0]))
    h, w, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for img in images:
        frame = cv2.imread(os.path.join(folder, img))
        out.write(frame)
    out.release()


def extract_frames_opencv(video_path: str, 
                         max_frames: Optional[int] = None,
                         start_frame: int = 0,
                         step: int = 1,
                         grayscale: bool = False) -> List[np.ndarray]:
    """
    Extract frames from video using OpenCV
    
    Args:
        video_path: Path to the MP4 video file
        max_frames: Maximum number of frames to extract (None for all)
        start_frame: Frame number to start from (0-indexed)
        step: Extract every nth frame (1 for every frame)
        grayscale: Convert to grayscale (useful for thermal imagery)
    
    Returns:
        List of frames as numpy arrays
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {width}x{height}")
    
    frames = []
    frame_count = 0
    extracted_count = 0
    
    # Skip to start frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_count = start_frame
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract frame based on step
        if (frame_count - start_frame) % step == 0:
            if grayscale:
                # Convert to grayscale for thermal imagery
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            frames.append(frame)
            extracted_count += 1
            
            # Check if we've reached max_frames
            if max_frames and extracted_count >= max_frames:
                break
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {len(frames)} frames")
    return frames


def extract_frames_with_timestamps(video_path: str,
                                 timestamps: List[float],
                                 grayscale: bool = False) -> List[np.ndarray]:
    """
    Extract frames at specific timestamps
    
    Args:
        video_path: Path to the MP4 video file
        timestamps: List of timestamps in seconds
        grayscale: Convert to grayscale
    
    Returns:
        List of frames at specified timestamps
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frames = []
    
    for timestamp in timestamps:
        # Convert timestamp to frame number
        frame_number = int(timestamp * fps)
        
        # Seek to specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if ret:
            if grayscale and len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
        else:
            print(f"Warning: Could not extract frame at timestamp {timestamp}s")
    
    cap.release()
    return frames

class VideoSynthesisSystem:
    """Main system for processing video frames and creating synthesized sequences"""
    
    def __init__(self, synthesizer: FrameSynthesizer):
        self.synthesizer = synthesizer
        self.current_path = os.path.abspath(os.path.dirname(__file__))
        self.project_root = os.path.abspath(os.path.join(self.current_path, ".."))
    
    def process_frame_sequence(self, frames: List[np.ndarray], 
                             num_intermediate: int = 1,
                             output_dir: Optional[str] = None) -> List[List[np.ndarray]]:
        """
        Process a sequence of frames to create synthesized training data
        
        Args:
            frames: List of input frames
            num_intermediate: Number of intermediate frames to synthesize between each pair
            output_dir: Optional directory to save synthesized sequences
            
        Returns:
            List of synthesized frame sequences, one for each consecutive frame pair
        """
        if len(frames) < 2:
            raise ValueError("Need at least 2 frames to synthesize intermediate frames")
        
        synthesized_sequences = []
        
        for i in range(len(frames) - 1):
            print(f"Processing frame pair {i+1}/{len(frames)-1}")
            
            frame1 = frames[i]
            frame2 = frames[i + 1]
            
            # Synthesize intermediate frames
            intermediate_frames = self.synthesizer.synthesize_frames(
                frame1, frame2, num_intermediate
            )
            
            # Create full sequence: frame1 + intermediates + frame2
            full_sequence = [frame1] + intermediate_frames + [frame2]
            synthesized_sequences.append(full_sequence)
            
            # Save sequence if output directory is provided
            if output_dir:
                self._save_sequence(full_sequence, i, output_dir)
        
        return synthesized_sequences
    
    def _save_sequence(self, sequence: List[np.ndarray], sequence_idx: int, output_dir: str):
        """Save a frame sequence to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        sequence_dir = output_path / f"sequence_{sequence_idx:04d}"
        sequence_dir.mkdir(exist_ok=True)
        
        for frame_idx, frame in enumerate(sequence):
            filename = sequence_dir / f"frame_{frame_idx:04d}.png"
            cv2.imwrite(str(filename), frame)
    
    def create_video_from_sequence(self, sequence: List[np.ndarray], 
                                 output_path: str, fps: int = 30):
        """Create a video file from a frame sequence"""
        if not sequence:
            return
        
        height, width = sequence[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in sequence:
            if len(frame.shape) == 2:  # Grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            out.write(frame)
        
        out.release()

    # Step is number of frames in each batch
    def prepare_milestone_frames(self, frames, cap_start, cap_end, fps, step):
        duration = cap_end - cap_start
        total_frames = int(duration * fps)
        milestone_indices = list(range(0, total_frames + 1, step))
        milestone_indices = [i for i in milestone_indices if i < len(frames)]
        # Extract milestone frames
        milestone_frames = [frames[i] for i in milestone_indices]
        return milestone_frames
    
    def extract_frames_between_timestamps(self, video_path: str,
                                          start_timestamp: float,
                                          end_timestamp: float,
                                          grayscale: bool = False,
                                          step: int = 1) -> List[np.ndarray]:
        """
        Extract ALL frames between two timestamps
        
        Args:
            video_path: Path to the MP4 video file
            start_timestamp: Start time in seconds (float)
            end_timestamp: End time in seconds (float)
            grayscale: Convert to grayscale
            step: Extract every nth frame (1 for all frames, 2 for every other frame, etc.)
        
        Returns:
            List of all frames between the timestamps
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if start_timestamp >= end_timestamp:
            raise ValueError("start_timestamp must be less than end_timestamp")
        
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps
        
        print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {video_duration:.2f}s duration")
        
        # Validate timestamps
        if end_timestamp > video_duration:
            print(f"Warning: end_timestamp ({end_timestamp}s) exceeds video duration ({video_duration:.2f}s)")
            end_timestamp = video_duration
        
        if start_timestamp < 0:
            print(f"Warning: start_timestamp ({start_timestamp}s) is negative, setting to 0")
            start_timestamp = 0
        
        # Convert timestamps to frame numbers
        start_frame = int(start_timestamp * fps)
        end_frame = int(end_timestamp * fps)
        
        print(f"Extracting frames {start_frame} to {end_frame} (timestamps {start_timestamp:.2f}s to {end_timestamp:.2f}s)")
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        current_frame_number = start_frame
        
        while current_frame_number <= end_frame:
            ret, frame = cap.read()
            
            if not ret:
                print(f"Warning: Could not read frame {current_frame_number}")
                break
            
            # Extract frame based on step
            if (current_frame_number - start_frame) % step == 0:
                if grayscale and len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                frames.append(frame)
            
            current_frame_number += 1
        
        cap.release()
        
        print(f"Successfully extracted {len(frames)} frames between {start_timestamp:.2f}s and {end_timestamp:.2f}s")
        return frames
    

    """
        Run it as:
        synthesizer = RAFTSynthesizer()
        # synthesizer = SimpleInterpolationSynthesizer()  # Alternative
        synthesis_system = VideoSynthesisSystem(synthesizer)
        synthesis_system.run_raft_example()
    """
    def run_raft_example(self):
        video_path = os.path.join(self.project_root, "dataset", "data", "1", "lynred_clipped.mp4")

        cap_start = 20.0
        cap_end = 23.0
        synthesis_freq = 0.1
        # fps
        fps = 60
        step = int(synthesis_freq * fps)

        # Create synthesizer (you can easily swap this for other methods)
        synthesizer = RAFTSynthesizer()
        # synthesizer = SimpleInterpolationSynthesizer()  # Alternative

        # Create synthesis system
        synthesis_system = VideoSynthesisSystem(synthesizer)

        frames = synthesis_system.extract_frames_between_timestamps(video_path, cap_start, cap_end)

        milestone_frames = synthesis_system.prepare_milestone_frames(frames, cap_start, cap_end, fps, step)

        # Process frames to create training data
        synthesized_sequences = synthesis_system.process_frame_sequence(
            milestone_frames, 
            num_intermediate= step - 2,  # Generate 3 frames between each pair
            output_dir="./synthesized_training_data"
        )

        print(f"Generated {len(synthesized_sequences)} synthesized sequences")
        merged_sequences = []

        # Optionally create videos from sequences
        for i, sequence in enumerate(synthesized_sequences):
            for frame in sequence:
                merged_sequences.append(frame)
        output_video_path = f"./synthesized_sequence.mp4"
        synthesis_system.create_video_from_sequence(merged_sequences, output_video_path, fps=60)