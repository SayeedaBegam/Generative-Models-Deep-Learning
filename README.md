# Generative Model — LoRA Style Fine-Tuning with Stable Diffusion 1.5
*“Ghibli Market”*

---

## Dataset

- **original/** : full-resolution PNG/JPG reference images  
- **512/** : 512 × 512 crops ready to train  
---

## Task Overview

1. **Tokenizer Setup**  
   Add a new style token `<sks>` to the tokenizer.  
   - Implemented in [`code/tokenization.py`](code/tokenization.py)  
   - Output: updated `tokenizer/` folder with vocab + embeddings  

2. **LoRA Fine-Tuning**  
   Finetune both the **UNet** and the **text encoder** with LoRA so that the prompt:  
   ```text
   a busy market, in <sks> style
---

##  What is Tokenization Here?

Stable Diffusion uses a **text tokenizer + encoder** (CLIP) to understand prompts.  
By default, it has no concept of your custom style (e.g., *Ghibli-like*).  

We add a new token (`<sks>`) and initialize its embedding so that during LoRA training,  
the model can **learn to associate `<sks>` with your dataset’s style**.

After this step, prompts like:

```
a busy market, in <sks> style
```

will be understood as "market scene + custom style placeholder".

---

##  Repository Structure

```
code/
  add_style_token.py    # script that adds <sks> to tokenizer + text encoder
  test_token.py        # quick verification that <sks> works
tokenizer/              # will be created after running add_style_token.py
```

---

##  Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Log in to Hugging Face

Make sure you have accepted the license for [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5).

```bash
hf auth login
```

---

## Usage

### Add the `<sks>` token

```bash
python code/add_style_token.py
```

This will:

- Add `<sks>` as a special token.
- Initialize its embedding using the phrase `"ghibli style"`.
- Save the updated tokenizer + text encoder to `tokenizer/`.

Output files include:

- `tokenizer/vocab.json`, `merges.txt`, etc.
- `tokenizer/text_encoder/model.safetensors`
- `tokenizer/token.json` (small manifest)

---

### Verify the token

```bash
python code/test_token.py
```
---

## Next Steps


---

## 📝 Notes

- Do **not** commit Hugging Face tokens.  
- The `tokenizer/` folder can be regenerated at any time by re-running the script.  
- LoRA training requires additional dependencies
