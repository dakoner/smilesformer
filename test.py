import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
import numpy as np

# --- 1. Vocabulary and Tokenizer ---
# We'll create a simple character-level vocabulary for SMILES strings.

class SMILES_Tokenizer:
    """
    A simple tokenizer for SMILES strings. It creates a vocabulary from a list
    of SMILES and converts strings to sequences of integers and back.
    """
    def __init__(self):
        # Special tokens
        self.pad_token = "<pad>"
        self.sos_token = "<sos>" # Start of Sequence
        self.eos_token = "<eos>" # End of Sequence
        self.unk_token = "<unk>" # Unknown token

        self.char_to_int = {}
        self.int_to_char = {}
        self.vocab_size = 0

    def fit(self, smiles_list):
        """
        Creates the vocabulary from a list of SMILES strings.
        """
        all_chars = set()
        for smiles in smiles_list:
            all_chars.update(list(smiles))

        # Sort for reproducibility
        sorted_chars = sorted(list(all_chars))

        # Add special tokens first
        special_tokens = [self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        for i, token in enumerate(special_tokens):
            self.char_to_int[token] = i
            self.int_to_char[i] = token

        # Add character tokens
        for i, char in enumerate(sorted_chars, start=len(special_tokens)):
            self.char_to_int[char] = i
            self.int_to_char[i] = char

        self.vocab_size = len(self.char_to_int)
        print(f"Vocabulary created. Size: {self.vocab_size}")

    def encode(self, smiles):
        """
        Converts a SMILES string to a list of integers, adding SOS and EOS tokens.
        """
        encoded = [self.char_to_int[self.sos_token]]
        for char in smiles:
            encoded.append(self.char_to_int.get(char, self.char_to_int[self.unk_token]))
        encoded.append(self.char_to_int[self.eos_token])
        return encoded

    def decode(self, int_sequence):
        """
        Converts a list of integers back to a SMILES string.
        Stops at the first EOS token.
        """
        chars = []
        for i in int_sequence:
            char = self.int_to_char.get(i)
            if char == self.eos_token:
                break
            if char not in [self.sos_token, self.pad_token]:
                chars.append(char)
        return "".join(chars)

# --- 2. Positional Encoding ---
# Transformers don't inherently understand sequence order, so we add positional encodings.

class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as
    the embeddings so that the two can be summed.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# --- 3. Transformer Autoencoder Model ---

class SmilesTransformer(nn.Module):
    """
    The main Transformer-based autoencoder model.
    """
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers,
                 num_decoder_layers, dim_feedforward, dropout=0.1):
        super(SmilesTransformer, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=False)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # Final linear layer to map to vocabulary size
        self.fc_out = nn.Linear(d_model, vocab_size)

        self.init_weights()

    def init_weights(self):
        """
        Initializes weights with a uniform distribution.
        """
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        """
        Generates a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        This prevents the decoder from "cheating" by looking at future tokens.
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt, src_padding_mask, tgt_padding_mask, device):
        """
        Forward pass for the autoencoder.
        Args:
            src: the sequence to the encoder (batch_size, seq_len)
            tgt: the sequence to the decoder (batch_size, seq_len)
            src_padding_mask: the mask for src keys (batch_size, seq_len)
            tgt_padding_mask: the mask for tgt keys (batch_size, seq_len)
        """
        # Note: PyTorch Transformer expects seq_len first (S, N, E)
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        # Embed and add positional encoding
        src_embed = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt_embed = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))

        # Generate decoder target mask
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(0)).to(device)

        # Encoder forward pass
        memory = self.transformer_encoder(src_embed, src_key_padding_mask=src_padding_mask)

        # Decoder forward pass
        output = self.transformer_decoder(tgt_embed, memory,
                                          tgt_mask=tgt_mask,
                                          tgt_key_padding_mask=tgt_padding_mask,
                                          memory_key_padding_mask=src_padding_mask)

        # Final output layer
        output = self.fc_out(output)

        # Transpose back to (batch_size, seq_len, vocab_size)
        return output.transpose(0, 1)


# --- 4. Training Setup ---

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
    D_MODEL = 128          # Embedding dimension
    NHEAD = 8              # Number of attention heads
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    DIM_FEEDFORWARD = 512  # Dimension of the feedforward network model in nn.TransformerEncoder
    DROPOUT = 0.1
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 32
    NUM_EPOCHS = 25

    # --- Data ---
    # Using a small, simple dataset for demonstration purposes.
    # In a real scenario, you would use a large dataset like ZINC or ChEMBL.
    smiles_data = [
        "CC(=O)OC1=CC=CC=C1C(=O)O", # Aspirin
        "C1=CC=C(C=C1)C(C(C(=O)O)N)O", # A random amino acid derivative
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", # Ibuprofen
        "C1=CC=C2C(=C1)C(=O)C3=C(C2=O)C=C(C=C3)N", # An aminoanthraquinone
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C4=CN=CC=C4", # Imatinib
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", # Caffeine
        "C(C(=O)O)N", # Glycine
        "CC(C(=O)O)N", # Alanine
        "C1=CN=CN1", # Imidazole
        "C1CCOC1", # Tetrahydrofuran
        "C1=CC=CS1", # Thiophene
        "CC(=O)NCCC1=CNC2=C1C=C(C=C2)OC", # Melatonin
    ]
    # Add more data for better training
    smiles_data *= 10 # Repeat data to simulate a larger dataset
    random.shuffle(smiles_data)

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
    model.eval()
    with torch.no_grad():
        test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O" # Aspirin
        print(f"Original SMILES: {test_smiles}")

        src = create_padded_batch([test_smiles], tokenizer, DEVICE)
        src_padding_mask = (src == tokenizer.char_to_int[tokenizer.pad_token])
        
        # Encode the source SMILES
        src_embed = model.pos_encoder(model.embedding(src.transpose(0, 1)) * math.sqrt(D_MODEL))
        memory = model.transformer_encoder(src_embed, src_key_padding_mask=src_padding_mask)

        # Start decoding with the <sos> token
        tgt_tokens = [tokenizer.char_to_int[tokenizer.sos_token]]
        max_len_decode = len(test_smiles) + 10

        for _ in range(max_len_decode):
            tgt_tensor = torch.LongTensor(tgt_tokens).unsqueeze(1).to(DEVICE) # (seq_len, 1)
            
            tgt_embed = model.pos_encoder(model.embedding(tgt_tensor) * math.sqrt(D_MODEL))
            tgt_mask = model._generate_square_subsequent_mask(tgt_tensor.size(0)).to(DEVICE)

            output = model.transformer_decoder(tgt_embed, memory, tgt_mask)
            output = model.fc_out(output)

            # Get the last token prediction
            next_token_logits = output[-1, 0, :]
            next_token_id = next_token_logits.argmax(0).item()
            
            if next_token_id == tokenizer.char_to_int[tokenizer.eos_token]:
                break
            
            tgt_tokens.append(next_token_id)

        reconstructed_smiles = tokenizer.decode(tgt_tokens)
        print(f"Reconstructed SMILES: {reconstructed_smiles}")
