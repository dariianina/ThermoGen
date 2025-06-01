import logging
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import os
import lpips

# --- Logging and TensorBoard ---
writer = SummaryWriter(log_dir="./runs/unet_cidis_V4_droneV1_l1+perc")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Paths and Hyperparameters ---
source_dir = "./dataset/drone_data/grayscale"  # update for your new dataset
target_dir = "./dataset/drone_data/thermal"    # update for your new dataset
checkpoint_dir = "./ckpt_ftdrone_l1+perc"
os.makedirs(checkpoint_dir, exist_ok=True)
batch_size = 64
num_epochs = 100
lr = 1e-3
image_size = (440, 258)

# --- Dataset ---
class PairedDataset(Dataset):
    def __init__(self, source_dir, target_dir, image_size=(256, 256)):
        self.source_paths = sorted(list(Path(source_dir).glob("*.png")) + list(Path(source_dir).glob("*.bmp")))
        self.target_paths = sorted(list(Path(target_dir).glob("*.png")) + list(Path(target_dir).glob("*.bmp")))
        self.transform = transforms.Compose([
            transforms.Resize((image_size[1], image_size[0])),
            transforms.ToTensor(),
        ])
    def __len__(self):
        return len(self.source_paths)
    def __getitem__(self, idx):
        src = Image.open(self.source_paths[idx]).convert("L")
        tgt = Image.open(self.target_paths[idx]).convert("L")
        src_tensor = self.transform(src)  # [1, H, W]
        tgt_tensor = self.transform(tgt)  # [1, H, W]
        return src_tensor, tgt_tensor

dataset = PairedDataset(source_dir, target_dir, image_size=image_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- Simple 1-channel U-Net ---
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

# --- Model, Loss, Optimizer ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.L1Loss()
lpips_loss = lpips.LPIPS(net='vgg').to(device)

# --- Load checkpoint for finetuning ---
finetune_ckpt_path = "./ckpt_unet_cidis_V4_l1+perc/unet_step_700.pt"  # <-- set your checkpoint path
if os.path.exists(finetune_ckpt_path):
    checkpoint = torch.load(finetune_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    logger.info(f"Loaded model weights from {finetune_ckpt_path}")
    # Optionally resume optimizer state:
    # optimizer.load_state_dict(checkpoint["optimizer"])
else:
    logger.info("No checkpoint found, training from scratch.")

# --- Validation Sample ---
validation_grayscale_path = "./dataset/drone_data/grayscale/000026.png"
validation_thermal_path = "./dataset/drone_data/thermal/000026.png"

validation_img = Image.open(validation_grayscale_path).convert("L")
validation_thermal_img = Image.open(validation_thermal_path).convert("L")

val_tensor = transforms.Compose([
    transforms.Resize((image_size[1], image_size[0])),
    transforms.ToTensor(),
])(validation_img).unsqueeze(0).to(device)  # [1, 1, H, W]

val_thermal_tensor = transforms.Compose([
    transforms.Resize((image_size[1], image_size[0])),
    transforms.ToTensor(),
])(validation_thermal_img).unsqueeze(0).to(device)

# --- Training Loop ---
global_step = 0
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        pred = model(src)
        l1 = criterion(pred, tgt)
        # LPIPS expects 3 channels, so repeat to [B, 3, H, W]
        pred_lpips = pred.repeat(1, 3, 1, 1)
        tgt_lpips = tgt.repeat(1, 3, 1, 1)
        perceptual = lpips_loss(pred_lpips, tgt_lpips).mean()
        total_loss = l1 + 0.1 * perceptual  # You can tune the weight
        total_loss.backward()
        optimizer.step()

        writer.add_scalar("Loss/train", total_loss.item(), global_step)
        epoch_loss += total_loss.item()
        num_batches += 1

        if global_step % 10 == 0:
            logger.info(f"Epoch {epoch} Step {global_step} Loss: {total_loss.item()}")

        # --- Save checkpoint and log validation every 100 steps ---
        if global_step % 100 == 1 and global_step > 0:
            ckpt_path = os.path.join(checkpoint_dir, f"unet_step_{global_step}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": global_step,
            }, ckpt_path)
            logger.info(f"Checkpoint saved at {ckpt_path}")

            # --- Inference and log image to TensorBoard ---
            model.eval()
            with torch.no_grad():
                val_pred = model(val_tensor)
                val_pred = val_pred.clamp(0, 1)
                writer.add_image("Validation/grayscale_input", val_tensor[0], global_step)
                writer.add_image("Validation/thermal_pred", val_pred[0], global_step)
                writer.add_image("Validation/thermal_gt", val_thermal_tensor[0], global_step)
            model.train()

        global_step += 1

    avg_epoch_loss = epoch_loss / num_batches
    writer.add_scalar("Loss/epoch_avg", avg_epoch_loss, epoch)
    logger.info(f"Epoch {epoch} Average Loss: {avg_epoch_loss}")

logger.info("Training complete.")
writer.close()