import math
import torch
import torch.nn as nn

from position_encoder import PositionalEncoding

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
