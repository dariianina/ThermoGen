import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path
import os
from itertools import chain

# --- Define U-Net (must match your training script) ---
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU())
        self.final = nn.Conv2d(32, 1, 1)
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.pool(x1)
        x3 = self.enc2(x2)
        x4 = self.up(x3)
        x5 = self.dec1(x4)
        out = self.final(x5)
        return out

# --- Config ---
ckpt_path = "./ckpt_ftdrone_l1+perc/unet_step_401.pt"  # <- set to your best checkpoint
output_dir = "./inference_v15_out"
os.makedirs(output_dir, exist_ok=True)
image_size = (440, 258)  # (width, height), match training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load model and weights ---
model = SimpleUNet().to(device)
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()

# --- Define transforms (must match training) ---
transform = transforms.Compose([
    transforms.Resize((image_size[1], image_size[0])),
    transforms.ToTensor(),
])

# --- Inference on a single image ---
def infer_one(img_path, save_path=None):
    img = Image.open(img_path).convert("L")
    inp = transform(img).unsqueeze(0).to(device)  # [1, 1, H, W]
    with torch.no_grad():
        pred = model(inp)
        pred = pred.clamp(0, 1)
    # Save result as image
    pred_img = transforms.ToPILImage()(pred[0].cpu())
    if save_path:
        pred_img.save(save_path)
    return pred_img

# --- Example: infer on a folder of grayscale images ---
test_dir = "./inference_v15"
image_paths = chain(
    Path(test_dir).glob("*.png"),
    Path(test_dir).glob("*.jpg"),
    Path(test_dir).glob("*.jpeg"),
)
print(image_paths)
for img_path in image_paths:
    print(img_path)
    out_name = os.path.basename(img_path)
    out_path = os.path.join(output_dir, out_name)
    infer_one(img_path, save_path=out_path)
    print(f"Inferred and saved: {out_path}")

print("Inference complete!")

