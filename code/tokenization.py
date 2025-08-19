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

from __future__ import annotations

import json
import os
from typing import List

import torch
from transformers import CLIPTextModel, CLIPTokenizer


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


def main() -> None:
    # Ensure output directory exists
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load tokenizer and text encoder from their subfolders
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        MODEL_NAME, subfolder="text_encoder", torch_dtype=torch.float16
    )

    # Add the new token, if not already present
    token_was_added = add_token_to_tokenizer(tokenizer, NEW_TOKEN)

    # If we added a token, we must resize the embedding matrix
    if token_was_added:
        text_encoder.resize_token_embeddings(len(tokenizer))
        print(f"Added {NEW_TOKEN} to tokenizer.")
    else:
        print(f"{NEW_TOKEN} already exists in the tokenizer.")

    # Initialize the new token's embedding
    token_id = tokenizer.convert_tokens_to_ids(NEW_TOKEN)
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise RuntimeError(
            f"Failed to resolve an ID for {NEW_TOKEN}. "
            "Did adding the special token succeed?"
        )

    init_vec = compute_initializer_embedding(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        initializer_text=INIT_TEXT,
    )

    with torch.no_grad():
        embedding_layer = text_encoder.get_input_embeddings()
        embedding_layer.weight[token_id] = init_vec

    # Save the updated tokenizer and text encoder for use in training
    tokenizer.save_pretrained(OUT_DIR)
    text_encoder.save_pretrained(os.path.join(OUT_DIR, "text_encoder"))

    # Save a tiny manifest for traceability
    manifest = {
        "token": NEW_TOKEN,
        "initializer": INIT_TEXT,
        "base_model": MODEL_NAME,
    }
    with open(os.path.join(OUT_DIR, "token.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"Done. Added {NEW_TOKEN} (id={token_id}). "
        f"Files written to '{OUT_DIR}'."
    )


if __name__ == "__main__":
    main()
