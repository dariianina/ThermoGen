import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import torchvision.transforms as transforms
from pathlib import Path

class FrameSynthesizer(ABC):
    """Abstract base class for frame synthesis methods"""
    
    @abstractmethod
    def synthesize_frames(self, frame1: np.ndarray, frame2: np.ndarray, 
                         num_intermediate: int = 1) -> List[np.ndarray]:
        """
        Synthesize intermediate frames between two input frames
        
        Args:
            frame1: First frame (numpy array)
            frame2: Second frame (numpy array)
            num_intermediate: Number of intermediate frames to generate
            
        Returns:
            List of synthesized frames
        """
        pass

class RAFTSynthesizer(FrameSynthesizer):
    """RAFT-based frame synthesizer using optical flow"""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu', 
                 use_fallback: bool = False):
        self.device = device
        self.use_fallback = use_fallback
        
        if use_fallback:
            print("Using simple interpolation fallback instead of RAFT")
            self.model = None
        else:
            self.model = self._load_raft_model()
            
        # If RAFT loading failed, use fallback
        if self.model is None:
            print("RAFT model unavailable, using interpolation fallback")
            self.fallback_synthesizer = SimpleInterpolationSynthesizer()
    
    def _fix_ssl_certificates(self):
        """Fix SSL certificate verification issues"""
        import ssl
        import urllib.request
        
        # Method 1: Disable SSL verification (not recommended for production)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
        urllib.request.install_opener(opener)
        
        # Method 2: Set environment variable (alternative)
        import os
        os.environ['TORCH_HOME'] = './torch_models'  # Local cache directory
    
    def _load_raft_model(self):
        """Load pre-trained RAFT model with multiple fallback strategies"""
        
        # Strategy 1: Fix SSL and try normal loading
        try:
            self._fix_ssl_certificates()
            
            import torchvision.models.optical_flow as flow_models
            print("Downloading RAFT model (first time may take several minutes)...")
            
            model = flow_models.raft_large(pretrained=True)
            model.to(self.device)
            model.eval()
            print("✓ RAFT model loaded successfully!")
            return model
            
        except Exception as e:
            print(f"✗ Strategy 1 failed: {e}")
        
        # Strategy 2: Try with different torchvision version
        try:
            import torchvision
            print(f"Torchvision version: {torchvision.__version__}")
            
            # For older torchvision versions
            if hasattr(torchvision.models, 'optical_flow'):
                import torchvision.models.optical_flow as flow_models
                model = flow_models.raft_large(weights='DEFAULT')  # New API
                model.to(self.device)
                model.eval()
                print("✓ RAFT model loaded with new API!")
                return model
                
        except Exception as e:
            print(f"✗ Strategy 2 failed: {e}")
        
        # Strategy 3: Manual model loading (if you have local weights)
        try:
            model_path = './models/raft_large.pth'
            if os.path.exists(model_path):
                import torchvision.models.optical_flow as flow_models
                model = flow_models.raft_large(pretrained=False)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.to(self.device)
                model.eval()
                print("✓ RAFT model loaded from local file!")
                return model
        except Exception as e:
            print(f"✗ Strategy 3 failed: {e}")
        
        print("✗ All RAFT loading strategies failed. Using interpolation fallback.")
        return None
    """RAFT-based frame synthesizer using optical flow"""
        
    def _load_raft_model(self):
        """Load pre-trained RAFT model with SSL certificate handling"""
        try:
            # Fix SSL certificate issues
            import ssl
            import urllib.request
            
            # Create unverified SSL context (temporary fix)
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Install the opener
            opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
            urllib.request.install_opener(opener)
            
            # Now try to load RAFT model
            import torchvision.models.optical_flow as flow_models
            
            print("Downloading RAFT model (this may take a while on first run)...")
            model = flow_models.raft_large(pretrained=True)
            model.to(self.device)
            model.eval()
            print("RAFT model loaded successfully!")
            return model
            
        except ImportError as e:
            print(f"Warning: torchvision optical flow models not available: {e}")
            print("Using simple interpolation fallback.")
            return None
        except Exception as e:
            print(f"Error loading RAFT model: {e}")
            print("Trying alternative loading methods...")
            
            # Alternative: Try loading without SSL verification
            try:
                import torchvision.models.optical_flow as flow_models
                # Set torch hub to use insecure downloads
                import torch
                torch.hub.set_dir('./models')  # Local directory for models
                
                model = flow_models.raft_large(pretrained=True)
                model.to(self.device)
                model.eval()
                return model
            except Exception as e2:
                print(f"Alternative loading also failed: {e2}")
                print("Using simple interpolation as fallback.")
                return None
    
    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Convert numpy frame to tensor format expected by RAFT"""
        if len(frame.shape) == 2:  # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif len(frame.shape) == 3 and frame.shape[2] == 1:  # Single channel
            frame = np.repeat(frame, 3, axis=2)
            
        # Normalize to [0, 1] and convert to tensor
        frame_tensor = torch.from_numpy(frame).float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # BCHW format
        return frame_tensor.to(self.device)
    
    def _compute_optical_flow(self, frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
        """Compute optical flow between two frames using RAFT"""
        if self.model is None:
            # Placeholder flow computation if RAFT model is not available
            h, w = frame1.shape[-2:]
            flow = torch.zeros(1, 2, h, w, device=self.device)
            return flow
            
        with torch.no_grad():
            flow = self.model(frame1, frame2)
            if isinstance(flow, list):
                flow = flow[-1]  # Take the final flow prediction
        return flow
    
    def _warp_frame(self, frame: torch.Tensor, flow: torch.Tensor, alpha: float) -> torch.Tensor:
        """Warp frame using optical flow scaled by alpha"""
        B, C, H, W = frame.shape
        
        # Scale flow by interpolation factor
        scaled_flow = flow * alpha
        
        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, H, device=self.device),
            torch.arange(0, W, device=self.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).float().unsqueeze(0)  # 1x2xHxW
        
        # Add flow to grid
        sampling_grid = grid + scaled_flow
        
        # Normalize to [-1, 1] for grid_sample
        sampling_grid[:, 0] = 2.0 * sampling_grid[:, 0] / (W - 1) - 1.0  # x
        sampling_grid[:, 1] = 2.0 * sampling_grid[:, 1] / (H - 1) - 1.0  # y
        
        # Transpose to HxWx2 format for grid_sample
        sampling_grid = sampling_grid.permute(0, 2, 3, 1)
        
        # Warp frame
        warped = F.grid_sample(frame, sampling_grid, align_corners=True, padding_mode='border')
        return warped
    
    def synthesize_frames(self, frame1: np.ndarray, frame2: np.ndarray, 
                         num_intermediate: int = 1) -> List[np.ndarray]:
        """Synthesize intermediate frames using RAFT or fallback to interpolation"""
        
        # If RAFT model is not available, use simple interpolation
        if self.model is None:
            print("Using interpolation fallback...")
            return self.fallback_synthesizer.synthesize_frames(frame1, frame2, num_intermediate)
        
        # Use RAFT synthesis
        return self._synthesize_with_raft(frame1, frame2, num_intermediate)
    
    def _synthesize_with_raft(self, frame1: np.ndarray, frame2: np.ndarray, 
                             num_intermediate: int = 1) -> List[np.ndarray]:
        """Original RAFT synthesis method"""
        # Preprocess frames
        tensor1 = self._preprocess_frame(frame1)
        tensor2 = self._preprocess_frame(frame2)
        
        # Compute bidirectional optical flow
        flow_12 = self._compute_optical_flow(tensor1, tensor2)  # frame1 -> frame2
        flow_21 = self._compute_optical_flow(tensor2, tensor1)  # frame2 -> frame1
        
        synthesized_frames = []
        
        for i in range(1, num_intermediate + 1):
            alpha = i / (num_intermediate + 1)
            
            # Warp both frames towards the target time
            warped1 = self._warp_frame(tensor1, flow_12, alpha)
            warped2 = self._warp_frame(tensor2, flow_21, 1 - alpha)
            
            # Blend warped frames
            intermediate = alpha * warped2 + (1 - alpha) * warped1
            
            # Convert back to numpy
            frame_np = intermediate.squeeze(0).permute(1, 2, 0).cpu().numpy()
            frame_np = (frame_np * 255).astype(np.uint8)
            
            # Convert back to original format if needed
            if len(frame1.shape) == 2:  # Original was grayscale
                frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)
            
            synthesized_frames.append(frame_np)
        
        return synthesized_frames

class SimpleInterpolationSynthesizer(FrameSynthesizer):
    """Simple linear interpolation synthesizer as fallback/baseline"""
    
    def synthesize_frames(self, frame1: np.ndarray, frame2: np.ndarray, 
                         num_intermediate: int = 1) -> List[np.ndarray]:
        """Simple linear interpolation between frames"""
        synthesized_frames = []
        
        for i in range(1, num_intermediate + 1):
            alpha = i / (num_intermediate + 1)
            interpolated = (1 - alpha) * frame1.astype(np.float32) + alpha * frame2.astype(np.float32)
            synthesized_frames.append(interpolated.astype(np.uint8))
        
        return synthesized_frames