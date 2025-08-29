import os, torch
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model
from peft.utils import set_peft_model_state_dict
from diffusers import StableDiffusionPipeline

# Config
MODEL_NAME     = "runwayml/stable-diffusion-v1-5"
INSTANCE_TOKEN = "<sks>"
OUTPUT_DIR     = "submission/lora_out"
device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pipeline
pipe = StableDiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device)
pipe.tokenizer.add_special_tokens({"additional_special_tokens": [INSTANCE_TOKEN]})
pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))

# Token ID check
token_id = pipe.tokenizer.convert_tokens_to_ids(INSTANCE_TOKEN)
print(f"[debug] token ID for {INSTANCE_TOKEN}: {token_id}")

# Load LoRA weights
lora_path = os.path.join(OUTPUT_DIR, "pytorch_lora_weights_sub.safetensors")
state_dict = load_file(lora_path)

# Restore token embedding if present
if "text_encoder.token_embedding.weight" in state_dict:
    with torch.no_grad():
        embedding = state_dict["text_encoder.token_embedding.weight"].to(pipe.text_encoder.dtype)
        pipe.text_encoder.get_input_embeddings().weight[token_id].copy_(embedding)
    print(f"[ok] restored token embedding for {INSTANCE_TOKEN}")
else:
    print(f"[warn] no token embedding found; {INSTANCE_TOKEN} will be random")

# Split LoRA weights
unet_sd = {k.replace("unet.", ""): v for k, v in state_dict.items() if k.startswith("unet.")}
txt_sd  = {k.replace("text_encoder.", ""): v for k, v in state_dict.items() if k.startswith("text_encoder.")}

# Attach adapters
unet_cfgs = LoraConfig(r=8, lora_alpha=16, target_modules=["to_q", "to_k", "to_v", "to_out.0"], bias="none")
txt_cfgs = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "out_proj"], bias="none")
pipe.unet = get_peft_model(pipe.unet, unet_cfgs)
pipe.text_encoder = get_peft_model(pipe.text_encoder, txt_cfgs)

set_peft_model_state_dict(pipe.unet, unet_sd)
set_peft_model_state_dict(pipe.text_encoder, txt_sd)

pipe.unet.set_adapter("default")
pipe.text_encoder.set_adapter("default")

# Memory optimizations
try:
    pipe.enable_attention_slicing()
    pipe.enable_model_cpu_offload()
except Exception as e:
    print(f"[warn] memory optimizations not available: {e}")

# Generate samples
os.makedirs("samples", exist_ok=True)
prompt = f"a busy market, in {INSTANCE_TOKEN} style"

# for i in range(3):
with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
    seeds = [1, 45, 50, 95, 500]
    for seed in seeds:
        g = torch.Generator(device=device).manual_seed(seed)
        # image = pipe(prompt,  generator=g, guidance_scale=7.5, negative_prompt="ugly, deformed, disfigured, poor details, bad anatomy").images[0]
        # image = pipe(prompt,  generator=g, guidance_scale=5.5, negative_prompt="ugly, deformed, disfigured, poor details, bad anatomy").images[0]
        image = pipe(prompt,  generator=g, guidance_scale=7.5, negative_prompt="blurry, low quality, curves, artifacts, ugly, low contrast, abnormal faces, text").images[0]
        # img = pipe(f"a busy market, in {INSTANCE_TOKEN} style", num_inference_steps=28, guidance_scale=5.5).images[0]
        image.save(os.path.join("submission/samples", f"sample_seed_{seed}.png"))
