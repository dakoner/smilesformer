import torch

import torch
import torch.nn as nn
import torch.optim as optim
from smiles_data import smiles_data
from smiles_tokenizer import SMILES_Tokenizer
from smiles_transformer import SmilesTransformer
from create_padded_batch import create_padded_batch
from constants import D_MODEL, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD, DROPOUT, LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS
from smiles_data import smiles_data

# --- 4. Training Setup ---


def train_epoch(model, optimizer, criterion, dataloader, tokenizer, device):
    """
    Performs one training epoch.
    """
    model.train()
    total_loss = 0
    pad_id = tokenizer.char_to_int[tokenizer.pad_token]

    for batch in dataloader:
        # Create padded tensors for the batch
        src = create_padded_batch(batch, tokenizer, device)
        
        # For an autoencoder, the target is the same as the source.
        # However, the decoder input should be shifted right (teacher forcing).
        # tgt_input starts with <sos> and ends before the last token.
        # tgt_output starts from the first token and ends with <eos>.
        tgt_input = src[:, :-1]
        tgt_output = src[:, 1:]

        # Create padding masks
        src_padding_mask = (src == pad_id)
        tgt_padding_mask = (tgt_input == pad_id)

        optimizer.zero_grad()

        # Forward pass
        output = model(src, tgt_input, src_padding_mask, tgt_padding_mask, device)

        # Calculate loss
        # We need to reshape output and target for CrossEntropyLoss
        # Output: (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
        # Target: (batch_size, seq_len) -> (batch_size * seq_len)
        loss = criterion(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # Gradient clipping
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# --- 5. Main Execution ---

if __name__ == '__main__':
    # --- Hyperparameters ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)
    

    # --- Data ---
    # Using a small, simple dataset for demonstration purposes.
    # In a real scenario, you would use a large dataset like ZINC or ChEMBL.
    
   

    # --- Initialization ---
    tokenizer = SMILES_Tokenizer()
    tokenizer.fit(smiles_data)

    model = SmilesTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(DEVICE)

    # Ignore padding index in loss calculation
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.char_to_int[tokenizer.pad_token])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Create a simple dataloader
    dataloader = [smiles_data[i:i + BATCH_SIZE] for i in range(0, len(smiles_data), BATCH_SIZE)]

    print(f"Starting training on {DEVICE}...")
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # --- Training Loop ---
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, optimizer, criterion, dataloader, tokenizer, DEVICE)
        print(f"Epoch: {epoch:02}, Train Loss: {train_loss:.4f}")

    # --- Inference/Reconstruction Example ---
    print("\n--- Reconstruction Test ---")
    torch.save(model.state_dict(), "model.save.pth")

