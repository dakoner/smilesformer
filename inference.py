import torch
import math
from create_padded_batch import create_padded_batch
from constants import D_MODEL

def inference(model, DEVICE, tokenizer, test_smiles):
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
    return reconstructed_smiles