import logging
import torch
from pathlib import Path
from diffusers import ControlNetModel, UNet2DConditionModel, AutoencoderKL
from transformers import AutoTokenizer, CLIPTextModel
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import os

writer = SummaryWriter(log_dir="./runs/run_overfit_1")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pretrained_model = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pretrained_controlnet = "lllyasviel/control_v11p_sd15_seg"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(pretrained_model, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(pretrained_model, subfolder="unet")
controlnet = ControlNetModel.from_pretrained(pretrained_controlnet)

optimizer = torch.optim.AdamW(controlnet.parameters(), lr=1e-5)

source_dir = "./dataset_1/grayscale"
target_dir = "./dataset_1/thermal"

class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, source_dir, target_dir, image_size=512):
        self.source_paths = sorted(list(Path(source_dir).glob("*.png")))
        self.target_paths = sorted(list(Path(target_dir).glob("*.png")))
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    def __len__(self):
        return len(self.source_paths)
    def __getitem__(self, idx):
        src = Image.open(self.source_paths[idx]).convert("RGB")
        tgt = Image.open(self.target_paths[idx]).convert("RGB")
        return {
            "conditioning_pixel_values": self.transform(src),
            "pixel_values": self.transform(tgt),
            "input_ids": tokenizer(
                "", return_tensors="pt", padding="max_length", truncation=True, max_length=tokenizer.model_max_length
            ).input_ids[0]
        }

dataset = PairedDataset(source_dir, target_dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = vae.to(device)
unet = unet.to(device)
controlnet = controlnet.to(device)
text_encoder = text_encoder.to(device)

# Prepare validation grayscale image for inference
validation_grayscale_path = "./dataset/grayscale/000026.png"
validation_img = Image.open(validation_grayscale_path).convert("RGB")
validation_tensor = dataset.transform(validation_img).unsqueeze(0).to(device)
val_input_ids = tokenizer(
    "", return_tensors="pt", padding="max_length", truncation=True, max_length=tokenizer.model_max_length
).input_ids.to(device)

checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

global_step = 0
controlnet.train()
for epoch in range(1001):  # set your number of epochs
    for batch in dataloader:
        optimizer.zero_grad()
        pixel_values = batch["pixel_values"].to(device)
        conditioning_pixel_values = batch["conditioning_pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)

        # 1. Encode target (thermal) image to latent space
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            encoder_hidden_states = text_encoder(input_ids)[0]

        # 2. Add noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (latents.shape[0],), device=latents.device).long()
        noisy_latents = latents + noise  # For real training, use a scheduler

        # 3. Forward pass through ControlNet
        down_block_res_samples, mid_block_res_sample = controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=conditioning_pixel_values,
            return_dict=False,
        )
        
        # 4. Predict noise with UNet
        model_pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=[s for s in down_block_res_samples],
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )[0]

        # 5. Compute loss (MSE between predicted and true noise)
        loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
        loss.backward()
        optimizer.step()

        # 6. Log loss to TensorBoard
        writer.add_scalar("Loss/train", loss.item(), global_step)
        logger.info(f"Step {global_step} Loss: {loss.item()}")

        # 7. Save checkpoint every 100 steps
        if global_step % 100 == 0 and global_step > 0:
            ckpt_path = os.path.join(checkpoint_dir, f"controlnet_step_{global_step}.pt")
            torch.save({
                "controlnet": controlnet.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": global_step,
            }, ckpt_path)
            logger.info(f"Checkpoint saved at {ckpt_path}")

            # --- Inference and log image to TensorBoard ---
            with torch.no_grad():
                val_encoder_hidden_states = text_encoder(val_input_ids)[0]
                val_latents = vae.encode(validation_tensor).latent_dist.sample()
                val_latents = val_latents * vae.config.scaling_factor
                val_noise = torch.zeros_like(val_latents)
                val_timesteps = torch.zeros((1,), dtype=torch.long, device=device)

                val_down_block_res_samples, val_mid_block_res_sample = controlnet(
                    val_latents,
                    val_timesteps,
                    encoder_hidden_states=val_encoder_hidden_states,
                    controlnet_cond=validation_tensor,
                    return_dict=False,
                )
                val_model_pred = unet(
                    val_latents,
                    val_timesteps,
                    encoder_hidden_states=val_encoder_hidden_states,
                    down_block_additional_residuals=[s for s in val_down_block_res_samples],
                    mid_block_additional_residual=val_mid_block_res_sample,
                    return_dict=False,
                )[0]
                val_denoised_latents = val_latents - val_model_pred
                val_image = vae.decode(val_denoised_latents / vae.config.scaling_factor).sample
                val_image = (val_image.clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1]
                val_image = val_image.cpu()
                writer.add_image("Validation/ground_truth", val_image[0], global_step)
                writer.add_image("Validation/thermal_pred", val_image[0], global_step)
                logger.info(f"Validation image logged at step {global_step}")

        global_step += 1

logger.info("Training complete.")
writer.close()