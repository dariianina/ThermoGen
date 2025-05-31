import os
import torch
import cv2
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.transforms.functional import to_tensor, resize
from . import RAFTSynthesizer

class VideoGen:
    def __init__(self, method = ""):
        current_path = os.path.abspath(os.path.dirname(__file__))
        project_root = os.path.abspath(os.path.join(current_path, ".."))
        data_path = os.path.join(project_root, "dataset", "video_generation")
        output_folder = os.path.join(data_path, "outputs")
        frames_folder = os.path.join(data_path, "frames")
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(frames_folder, exist_ok=True)

        if method == "raft":
            self.run = self._run_raft
        elif method == "method2":
            self.run = self._run_method2
        else:
            raise ValueError(f"Unknown mode: {method}")

    # RAFT will fill the frames between the frames to create videos
    def _run_raft(self, frames_list):
        """ raft = RAFTSynthesizer()
        for idx, frame in enumerate(frames_list):
            next_frame = frames_list[idx + 1] if idx != len(frames_list) - 1 else None
            if next_frame:
                syntesized_frames = raft.synthesize_frames(frame, next_frame, 30) """
        pass
        

    def _interpolate_frame(img1, img2, flow_fw):
        """ Warp img1 halfway toward img2 using forward flow """
        B, C, H, W = img1.shape
        # Normalize pixel coordinates to [-1, 1]
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=-1).float().to(img1.device)  # H x W x 2
        grid = grid.unsqueeze(0)  # Add batch
        flow_half = flow_fw.permute(0, 2, 3, 1) * 0.5
        coords = grid + flow_half  # Forward half-way
        # Normalize to [-1, 1]
        coords[..., 0] = 2.0 * coords[..., 0] / (W - 1) - 1.0
        coords[..., 1] = 2.0 * coords[..., 1] / (H - 1) - 1.0
        return torch.nn.functional.grid_sample(img1, coords, align_corners=True)
    

    def _run_method2(self):
        print("Running in method 2")