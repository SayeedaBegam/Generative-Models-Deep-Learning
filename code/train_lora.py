import os, random, math
from pathlib import Path
from typing import List
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDPMScheduler, StableDiffusionPipeline, EulerAncestralDiscreteScheduler, \
                      EulerDiscreteScheduler, LMSDiscreteScheduler, DPMSolverMultistepScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from peft import LoraConfig, get_peft_model
from torchvision import transforms as T
from __future__ import annotations
import json
import os
from typing import List
import torch
from transformers import CLIPTextModel, CLIPTokenizer



# ---- Config (feel free to edit) ----
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
INSTANCE_TOKEN = "<sks>"
DATA_ROOT = "dataset"           # contains '512/' subfolder with training crops
IMAGES_SUBDIR = "512"
OUTPUT_DIR = "lora_out"
RESOLUTION = 512
LORA_RANK = 8
LR = 1e-4
MAX_STEPS = 800
BATCH_SIZE = 4
GRAD_ACCUM = 1
SEED = 42
MIXED_PRECISION = "fp16"  # "fp16" | "bf16" | "no"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("samples", exist_ok=True)

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## DATA LOADER
class ImageOnlyDataset(Dataset):
    def __init__(self, image_dir: str, prompt: str, resolution: int = 512):
        self.paths = []
        for ext in ('*.png','*.jpg','*.jpeg','*.webp','*.bmp'):
            self.paths.extend(sorted(Path(image_dir).glob(ext)))
        if len(self.paths) == 0:
            raise FileNotFoundError(f'No images found under {image_dir}')
        self.prompt = prompt
        self.tf = T.Compose([
            T.Resize((resolution, resolution), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.ToTensor(),
            T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]) # [-1,1]
        ])

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        im = Image.open(p).convert('RGB')
        px = self.tf(im)
        return {'pixel_values': px, 'prompt': self.prompt}

train_dir = os.path.join(DATA_ROOT, IMAGES_SUBDIR)
prompt = f'a busy market, in {INSTANCE_TOKEN} style'
dataset = ImageOnlyDataset(train_dir, prompt, resolution=RESOLUTION)

def collate(batch):
    pixel_values = torch.stack([b['pixel_values'] for b in batch])
    prompts = [b['prompt'] for b in batch]
    return {'pixel_values': pixel_values, 'prompts': prompts}

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                    collate_fn=collate, drop_last=True, pin_memory=True)
len(dataset), next(iter(loader))['pixel_values'].shape

# Pipeline and scheduler declaration

pipeline = StableDiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
pipeline.to("cuda")

# Choose ONE of the multiple schedulers:

# 1) Euler Ancestral (crisp, more texture/variety)
#pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)

# 2) Euler (smooth, painterly)
# pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)

# 3) LMS (stable, consistent detail)
# pipeline.scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)

# 4) DPM-Solver++ 2M (quality at low steps)
pipeline.scheduler = DPMSolverMultistepScheduler.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="scheduler",
    algorithm_type="dpmsolver++", solver_order=2, use_karras_sigmas=True
)

### SAYEEDA Tokenizer

"""
Add a new *style* token (e.g., "<sks>") to the Stable Diffusion 1.5
CLIP tokenizer and initialize its embedding so it can be trained later
(e.g., via LoRA). This script does **not** use your image data; it only
modifies the text tokenizer and text encoder.

Outputs (written to OUT_DIR):
- tokenizer files: vocab.json, merges.txt, tokenizer_config.json, special_tokens_map.json
- text encoder (updated embeddings): text_encoder/model.safetensors, text_encoder/config.json
- a small token.json manifest describing what was added

Prerequisites:
- You must be logged in to Hugging Face and have accepted the license
  for "runwayml/stable-diffusion-v1-5".
  Run: `huggingface-cli login` then paste the token value.

Run:
    python code/tokenization.py
"""


# ---------------- Configuration ----------------
MODEL_NAME: str = "runwayml/stable-diffusion-v1-5"
NEW_TOKEN: str = "<sks>"                # the style token you are adding
INIT_TEXT: str = "ghibli style"         # initializer phrase used to seed the new embedding
OUT_DIR: str = "tokenizer"              # where updated tokenizer & text encoder will be saved
# ------------------------------------------------


def add_token_to_tokenizer(tokenizer: CLIPTokenizer, token: str) -> bool:
    """
    Add `token` to `tokenizer` as an *additional special token* (kept as a single piece).

    Returns:
        True if the token was added; False if it already existed.
    """
    vocab = tokenizer.get_vocab()
    if token in vocab:
        return False

    tokenizer.add_special_tokens({"additional_special_tokens": [token]})
    return True


def compute_initializer_embedding(
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    initializer_text: str,
) -> torch.Tensor:
    """
    Compute an initial embedding for the new token by averaging the embeddings
    of the tokens produced by `initializer_text`.
    """
    with torch.no_grad():
        embedding_layer = text_encoder.get_input_embeddings()  # nn.Embedding
        init_ids: List[int] = tokenizer(
            initializer_text, add_special_tokens=False
        ).input_ids

        if len(init_ids) == 0:
            raise ValueError(
                "Initializer text produced no tokens. "
                "Choose a different INIT_TEXT."
            )

        init_vec = embedding_layer.weight[init_ids].mean(dim=0)
    return init_vec


os.makedirs(OUT_DIR, exist_ok=True)

tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(
    MODEL_NAME, subfolder="text_encoder", torch_dtype=torch.float16
)


token_was_added = add_token_to_tokenizer(tokenizer, NEW_TOKEN)
if token_was_added:
    text_encoder.resize_token_embeddings(len(tokenizer))
    print(f"Added {NEW_TOKEN} to tokenizer.")
else:
    print(f"{NEW_TOKEN} already exists in the tokenizer.")

token_id = tokenizer.convert_tokens_to_ids(NEW_TOKEN)
if token_id is None or token_id == tokenizer.unk_token_id:
    raise RuntimeError(f"Failed to resolve an ID for {NEW_TOKEN}.")

init_vec = compute_initializer_embedding(text_encoder, tokenizer, INIT_TEXT)

with torch.no_grad():
    emb = text_encoder.get_input_embeddings()
    emb.weight[token_id] = init_vec.to(emb.weight.dtype)

tokenizer.save_pretrained(OUT_DIR)                       # saves to OUT_DIR/tokenizer.json, etc.
text_encoder.save_pretrained(os.path.join(OUT_DIR, "text_encoder"))

# Plug in the updated tokenizer & text encoder
pipeline.tokenizer = CLIPTokenizer.from_pretrained(OUT_DIR)
pipeline.text_encoder = CLIPTextModel.from_pretrained(
    os.path.join(OUT_DIR, "text_encoder"),
    torch_dtype=torch.float16
).to("cuda")


### NOAS Lora Fine Tuner
from peft import LoraConfig, get_peft_model

# U_NET LoRA
unet_lora_cfg = LoraConfig(
    r=LORA_RANK,
    lora_alpha=2 * LORA_RANK,
    lora_dropout=0.05,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    bias="none",
)
pipeline.unet = get_peft_model(pipeline.unet, unet_lora_cfg)

# text encoder LoRA 
txt_lora_cfg = LoraConfig(
    r=LORA_RANK,
    lora_alpha=2 * LORA_RANK,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    bias="none",
)

pipeline.text_encoder = get_peft_model(pipeline.text_encoder, txt_lora_cfg)

# 3) Freeze base weights (PEFT does this, but be explicit)
for n,p in pipeline.unet.named_parameters():
    p.requires_grad = "lora_" in n or "lora_A" in n or "lora_B" in n
for n,p in pipeline.text_encoder.named_parameters():
    p.requires_grad = "lora_" in n or "lora_A" in n or "lora_B" in n

# 4) Optimizer collects only adapter params
train_params = [p for p in pipeline.unet.parameters() if p.requires_grad] + \
               [p for p in pipeline.text_encoder.parameters() if p.requires_grad]
optim = torch.optim.AdamW(train_params, lr=1e-4)



# ---- Build pipelineline + attach LoRA to UNet & text encoder ----

# Sanity Test: the special token must resolve to a real id
assert pipeline.tokenizer.convert_tokens_to_ids(INSTANCE_TOKEN) != pipeline.tokenizer.unk_token_id, \
    f"{INSTANCE_TOKEN} not found in tokenizer. Add/resize/init it first."

# Attach LoRA to UNet
unet_lora_cfg = LoraConfig(
    r=LORA_RANK, lora_alpha=2*LORA_RANK, lora_dropout=0.05,
    target_modules=["to_q","to_k","to_v","to_out.0"], bias="none"
)
pipeline.unet = get_peft_model(pipeline.unet, unet_lora_cfg)

# Attach LoRA to text encoder (CLIP)
txt_lora_cfg = LoraConfig(
    r=LORA_RANK, lora_alpha=2*LORA_RANK, lora_dropout=0.1,
    target_modules=["q_proj","k_proj","v_proj","out_proj"], bias="none"
)
pipeline.text_encoder = get_peft_model(pipeline.text_encoder, txt_lora_cfg)

# Make sure only LoRA params require grad (base weights frozen by PEFT)
def trainable_params(m):
    return [p for p in m.parameters() if p.requires_grad]

train_params = trainable_params(pipeline.unet) + trainable_params(pipeline.text_encoder)
optimizer = torch.optim.AdamW(train_params, lr=LR, betas=(0.9, 0.999), weight_decay=1e-2)

# ---- Training noise scheduler (matches SD1.5 pretraining) ----
noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
    prediction_type="epsilon"
)

scaler = torch.cuda.amp.GradScaler(enabled=(MIXED_PRECISION=="fp16"))
autocast_ctx = (torch.cuda.amp.autocast(dtype=torch.float16) if MIXED_PRECISION=="fp16"
                else (torch.cuda.amp.autocast(dtype=torch.bfloat16) if MIXED_PRECISION=="bf16"
                else nullcontext()))

# ---- Helper: encode images -> latents ----
def encode_vae(images_fp16):
    # images are [-1,1] float16 on CUDA; VAE expects same dtype on GPU here
    posterior = pipeline.vae.encode(images_fp16).latent_dist
    latents = posterior.sample() * 0.18215  # SD1.x scaling
    return latents

# ---- BEFORE sample (same seed used after training) ----
def generate_sample(name, seed=1234):
    g = torch.Generator(device=device).manual_seed(seed)
    # Use a nicer inference scheduler; training scheduler is independent
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    with torch.inference_mode(), autocast_ctx:
        img = pipeline(
            prompt=f"a busy market, in {INSTANCE_TOKEN} style",
            num_inference_steps=28, guidance_scale=5.5, generator=g
        ).images[0]
    img.save(os.path.join("samples", f"{name}.png"))


# ---- TRAINING LOOP  ----
pipeline.unet.train(); pipeline.text_encoder.train(); pipeline.vae.eval()  # VAE frozen
global_step = 0
grad_accum = max(1, GRAD_ACCUM)

pbar = tqdm(loader, desc="Training (1 epoch)", total=len(loader))
for step, batch in enumerate(pbar):
    pixel_values = batch["pixel_values"].to(device=device, dtype=pipeline.vae.dtype, non_blocking=True)

    # --- latents + noise ---
    with torch.no_grad():
        latents = encode_vae(pixel_values)
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (bsz,), device=device, dtype=torch.long
        )
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # --- text conditioning with 10% dropout (CFG emulation) ---
    prompts = batch["prompts"]
    # randomly drop text 10% of the time to improve robustness
    if random.random() < 0.10:
        cond_ids = pipeline.tokenizer(
            [""] * bsz, padding="max_length", truncation=True, max_length=pipeline.tokenizer.model_max_length, return_tensors="pt"
        ).input_ids.to(device)
    else:
        cond_ids = pipeline.tokenizer(
            prompts, padding="max_length", truncation=True, max_length=pipeline.tokenizer.model_max_length, return_tensors="pt"
        ).input_ids.to(device)

    with autocast_ctx:
        cond_emb = pipeline.text_encoder(cond_ids)[0]                      # (B, 77, 768)
        model_pred = pipeline.unet(noisy_latents, timesteps, cond_emb).sample  # predict ε
        loss = torch.nn.functional.mse_loss(model_pred, noise, reduction="mean")

    # --- optimize (with grad accumulation) ---
    if MIXED_PRECISION=="fp16":
        scaler.scale(loss / grad_accum).backward()
    else:
        (loss / grad_accum).backward()

    if (step + 1) % grad_accum == 0:
        if MIXED_PRECISION=="fp16":
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

    pbar.set_postfix({"loss": f"{loss.item():.4f}"})

# ---- AFTER sample (same seed) ----
pipeline.unet.eval(); pipeline.text_encoder.eval()

# ---- Save LoRA adapters ----
pipeline.unet.save_pretrained(os.path.join(OUTPUT_DIR, "lora_unet"))
pipeline.text_encoder.save_pretrained(os.path.join(OUTPUT_DIR, "lora_text"))
print("Saved LoRA adapters to:", OUTPUT_DIR)


