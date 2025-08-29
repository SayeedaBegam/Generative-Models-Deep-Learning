import os, torch, random
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import get_peft_model, LoraConfig
from peft.utils import get_peft_model_state_dict
from safetensors.torch import save_file

# onfig
MODEL_NAME     = "runwayml/stable-diffusion-v1-5"
INSTANCE_TOKEN = "<sks>"
DATA_DIR       = "submission/data/512"
RESOLUTION     = 512
LORA_RANK      = 8
LR             = 5e-5
MAX_STEPS      = 844
BATCH_SIZE     = 4
GRAD_ACCUM     = 1
device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR     = "submission/lora_out"

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset
class ImageDataset(Dataset):
    def __init__(self, folder):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith((".png", ".jpg"))]
        self.transform = transforms.Compose([
            transforms.Resize((RESOLUTION, RESOLUTION)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return {"pixel_values": self.transform(img), "prompts": f"{INSTANCE_TOKEN}"}
        # return {"pixel_values": self.transform(img), "prompts": f"a busy market, in {INSTANCE_TOKEN} style"}


loader = DataLoader(ImageDataset(DATA_DIR), batch_size=BATCH_SIZE, shuffle=True)

# Load pipeline
pipe = StableDiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
pipe.tokenizer.add_tokens([INSTANCE_TOKEN])
pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
pipe.to(device)

# Attach LoRA
unet_cfgs = LoraConfig(r=LORA_RANK, lora_alpha=2*LORA_RANK, target_modules=["to_q","to_k","to_v","to_out.0"], bias="none")
txt_cfgs = LoraConfig(r=LORA_RANK, lora_alpha=2*LORA_RANK, target_modules=["q_proj","k_proj","v_proj","out_proj"], bias="none")
pipe.unet = get_peft_model(pipe.unet, unet_cfgs)
pipe.text_encoder = get_peft_model(pipe.text_encoder, txt_cfgs)

# Optimizer
params = list(p for p in pipe.unet.parameters() if p.requires_grad) + list(p for p in pipe.text_encoder.parameters() if p.requires_grad)
optimizer = torch.optim.AdamW(params, lr=LR)

# Scheduler
# noise_scheduler = DDPMScheduler(num_train_timesteps=500, beta_start=0.0003, beta_end=0.002)
# noise_scheduler = DDPMScheduler(num_train_timesteps=500, beta_start=0.008, beta_end=0.001)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")

# Training
global_step = 0
pipe.unet.train()
pipe.text_encoder.train()
pipe.vae.eval()

for epoch in range(999):
    for batch in tqdm(loader, desc=f"Step {global_step}"):
        pixel_values = batch["pixel_values"].to(device, dtype=pipe.vae.dtype)
        with torch.no_grad():
            latents = pipe.vae.encode(pixel_values).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        if random.random() < 0.1:
            input_ids = pipe.tokenizer([""] * latents.shape[0], padding="max_length", truncation=True, max_length=pipe.tokenizer.model_max_length, return_tensors="pt").input_ids.to(device)
        else:
            input_ids = pipe.tokenizer(batch["prompts"], padding="max_length", truncation=True, max_length=pipe.tokenizer.model_max_length, return_tensors="pt").input_ids.to(device)

        encoder_hidden_states = pipe.text_encoder(input_ids)[0]
        noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        loss.backward()
        if global_step % GRAD_ACCUM == 0:
            optimizer.step(); optimizer.zero_grad()

        global_step += 1
        if global_step >= MAX_STEPS:
            # Extract and merge LoRA weights
            unet_sd = get_peft_model_state_dict(pipe.unet)
            txt_sd  = get_peft_model_state_dict(pipe.text_encoder)
            merged = {f"unet.{k}": v for k, v in unet_sd.items()}
            merged.update({f"text_encoder.{k}": v for k, v in txt_sd.items()})

            # Save token embedding for <sks>
            token_id = pipe.tokenizer.convert_tokens_to_ids(INSTANCE_TOKEN)
            embedding = pipe.text_encoder.get_input_embeddings().weight[token_id].detach().cpu()
            merged["text_encoder.token_embedding.weight"] = embedding

            # Save to file
            save_file(merged, os.path.join(OUTPUT_DIR, "pytorch_lora_weights.safetensors"))
            print(f"Training completed step {global_step} weights and token embedding saved")
            exit()

