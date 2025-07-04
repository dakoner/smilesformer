import torch
import math
from smiles_tokenizer import SMILES_Tokenizer
from smiles_transformer import SmilesTransformer
from constants import D_MODEL, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD, DROPOUT
from smiles_data import smiles_data
from create_padded_batch import create_padded_batch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

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
model.load_state_dict(torch.load("model.save.pth", weights_only=True))
model.eval()
with torch.no_grad():
    for test_smiles in [
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
    ]:

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
        print(f"Original SMILES: {test_smiles}")
        print(f"Reconstructed SMILES: {reconstructed_smiles}")
