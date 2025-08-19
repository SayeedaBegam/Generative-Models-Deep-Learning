from transformers import CLIPTokenizer
tok = CLIPTokenizer.from_pretrained("tokenizer")
text = "a busy market, in <sks> style"
ids = tok(text, add_special_tokens=False).input_ids
toks = tok.convert_ids_to_tokens(ids)
print("Tokenized:", toks)
print("Contains <sks>:", "<sks>" in toks)
