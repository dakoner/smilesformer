import torch

def create_padded_batch(batch_smiles, tokenizer, device):
    """
    Takes a list of SMILES strings, tokenizes them, and creates padded tensors.
    """
    pad_id = tokenizer.char_to_int[tokenizer.pad_token]
    batch_tokens = [tokenizer.encode(s) for s in batch_smiles]
    max_len = max(len(s) for s in batch_tokens)

    padded_batch = []
    for tokens in batch_tokens:
        padding = [pad_id] * (max_len - len(tokens))
        padded_batch.append(tokens + padding)

    return torch.tensor(padded_batch, dtype=torch.long, device=device)