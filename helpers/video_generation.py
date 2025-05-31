import os
import torch
import cv2
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.transforms.functional import to_tensor, resize

import numpy as np

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import random
from PIL import Image
import torchvision.transforms as transforms


class ThermalVideoProcessor:
    """Process thermal videos and extract frame pairs for training"""
    
    def __init__(self, video_path):
        self.video_path = video_path
        self.frames = []
        
    def extract_frames(self, max_frames=None):
        """Extract all frames from video"""
        cap = cv2.VideoCapture(self.video_path)
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to grayscale if needed (thermal videos often single channel)
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
            frames.append(frame)
            
            if max_frames and len(frames) >= max_frames:
                break
                
        cap.release()
        self.frames = frames
        print(f"Extracted {len(frames)} frames")
        return frames
    
    def create_training_pairs(self, gap_range=(40, 50)):
        """Create frame pairs with specified gap for training"""
        pairs = []
        
        for i in range(len(self.frames) - max(gap_range)):
            gap = random.randint(gap_range[0], gap_range[1])
            if i + gap < len(self.frames):
                pairs.append({
                    'frame_a': self.frames[i],
                    'frame_b': self.frames[i + gap],
                    'gap': gap,
                    'start_idx': i,
                    'end_idx': i + gap
                })
                
        print(f"Created {len(pairs)} training pairs")
        return pairs

class ThermalFrameDataset(Dataset):
    """Dataset for thermal frame interpolation"""
    
    def __init__(self, frame_pairs, img_size=(256, 256)):
        self.pairs = frame_pairs
        self.img_size = img_size
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        frame_a = self.transform(pair['frame_a'])
        frame_b = self.transform(pair['frame_b'])
        
        return {
            'frame_a': frame_a,
            'frame_b': frame_b,
            'gap': pair['gap']
        }

class ThermalFrameGenerator(nn.Module):
    """Lightweight neural network for thermal frame interpolation"""
    
    def __init__(self):
        super().__init__()
        
        # Encoder for both input frames
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 64, 7, 1, 3),  # 2 input channels (frame A + frame B)
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # Downsample
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),  # Downsample
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1),  # Downsample
            nn.ReLU(),
        )
        
        # Decoder to generate intermediate frame
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # Upsample
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # Upsample
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # Upsample
            nn.ReLU(),
            nn.Conv2d(64, 1, 7, 1, 3),  # Output single channel
            nn.Tanh()
        )
        
    def forward(self, frame_a, frame_b):
        # Concatenate input frames
        x = torch.cat([frame_a, frame_b], dim=1)
        
        # Encode
        encoded = self.encoder(x)
        
        # Decode to intermediate frame
        output = self.decoder(encoded)
        
        return output

class ThermalVideoSynthesizer:
    """Main class to synthesize thermal drone videos"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = ThermalFrameGenerator().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
    def train(self, dataloader, epochs=10):
        """Train the frame interpolation model"""
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch_idx, batch in enumerate(dataloader):
                frame_a = batch['frame_a'].to(self.device)
                frame_b = batch['frame_b'].to(self.device)
                
                # Generate intermediate frame
                pred_frame = self.model(frame_a, frame_b)
                
                # Target is simple average (you can improve this)
                target_frame = (frame_a + frame_b) / 2
                
                # Calculate loss
                loss = self.criterion(pred_frame, target_frame)
                
                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            
            avg_loss = total_loss / len(dataloader)
            print(f'Epoch {epoch} completed, Average Loss: {avg_loss:.4f}')
    
    def generate_intermediate_frames(self, frame_a, frame_b, num_frames=5):
        """Generate intermediate frames between two input frames"""
        self.model.eval()
        
        with torch.no_grad():
            # Convert frames to tensors
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            
            frame_a_tensor = transform(frame_a).unsqueeze(0).to(self.device)
            frame_b_tensor = transform(frame_b).unsqueeze(0).to(self.device)
            
            intermediate_frames = []
            
            # Generate frames by interpolating between A and B
            for i in range(1, num_frames + 1):
                alpha = i / (num_frames + 1)
                
                # Weighted interpolation in feature space
                blended_a = (1 - alpha) * frame_a_tensor
                blended_b = alpha * frame_b_tensor
                
                generated_frame = self.model(blended_a, blended_b)
                
                # Convert back to numpy
                frame_np = generated_frame.cpu().squeeze().numpy()
                frame_np = ((frame_np + 1) * 127.5).astype(np.uint8)  # Denormalize
                
                intermediate_frames.append(frame_np)
            
            return intermediate_frames
    
    def synthesize_video_sequence(self, start_frame, end_frame, sequence_length=50):
        """Create a full synthetic video sequence"""
        frames = self.generate_intermediate_frames(
            start_frame, end_frame, sequence_length - 2
        )
        
        # Add start and end frames
        full_sequence = [start_frame] + frames + [end_frame]
        
        return full_sequence
    
    def save_synthetic_video(self, frame_sequence, output_path, fps=30):
        """Save synthesized frames as video"""
        height, width = frame_sequence[0].shape
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), False)
        
        for frame in frame_sequence:
            out.write(frame)
        
        out.release()
        print(f"Synthetic video saved to {output_path}")

    # Quick usage example for hackathon
    def quick_thermal_synthesis(self, video_path, output_dir="synthetic_thermal"):
        """Fast pipeline for hackathon use"""
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Process video
        processor = ThermalVideoProcessor(video_path)
        frames = processor.extract_frames(max_frames=1000)  # Limit for speed
        
        # Create training pairs
        pairs = processor.create_training_pairs(gap_range=(40, 50))
        
        # Create dataset and dataloader
        dataset = ThermalFrameDataset(pairs[:100])  # Use subset for speed
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Initialize synthesizer
        synthesizer = ThermalVideoSynthesizer()
        
        # Quick training (reduce epochs for hackathon)
        print("Training model...")
        synthesizer.train(dataloader, epochs=5)
        
        # Generate synthetic videos
        print("Generating synthetic videos...")
        
        for i in range(3):  # Generate 3 synthetic videos
            # Pick random frame pairs
            start_idx = random.randint(0, len(frames) - 100)
            end_idx = start_idx + random.randint(40, 80)
            
            start_frame = frames[start_idx]
            end_frame = frames[end_idx]
            
            # Generate sequence
            synthetic_sequence = synthesizer.synthesize_video_sequence(
                start_frame, end_frame, sequence_length=50
            )
            
            # Save video
            output_path = f"{output_dir}/synthetic_thermal_{i+1}.mp4"
            synthesizer.save_synthetic_video(synthetic_sequence, output_path)
        
        print(f"Generated 3 synthetic thermal videos in {output_dir}/")
        return synthesizer

    # Trains the model with thermal vid and then generates synthetic videos
    def run_thermal_video_generator_example(self):
        # Replace with your thermal video path
        file_dir = os.path.abspath(os.path.dirname(__file__))
        project_root = os.path.join(file_dir, "..")
        video_path = os.path.join(project_root, "dataset", "data", "1", "lynred_clipped.mp4")
        
        # Run quick synthesis
        synthesizer = self.quick_thermal_synthesis(video_path)
        
        print("Thermal video synthesis completed!")
        print("You now have synthetic thermal drone interception videos for training!")