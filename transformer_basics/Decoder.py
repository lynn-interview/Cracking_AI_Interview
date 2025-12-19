import torch
import torch.nn as nn
import math

try:
    from .MultiHeadAttention import MultiHeadAttention
    from .DecoderBlock import DecoderBlock
except ImportError:  # pragma: no cover
    from MultiHeadAttention import MultiHeadAttention
    from DecoderBlock import DecoderBlock

class Decoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ffn_dim, vocab_size, max_seq_len, pad_token_id=0, dropout=0.1):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # A stack of DecoderBlocks
        self.layers = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)
        ])
        
        self.final_linear = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(embed_dim)

    def _create_causal_mask(self, seq_len, device):
        """
        Create causal mask (look-ahead mask)
        Returns: (1, T, T)
        """
        # torch.tril creates a lower triangular matrix
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
        # Add a 'batch' dimension for broadcasting
        return mask.unsqueeze(0) 

    def _create_padding_mask(self, input_ids, pad_token_id):
        """
        Create padding mask
        Returns: (B, 1, T)
        """
        mask = (input_ids != pad_token_id).unsqueeze(1).bool()
        return mask

    def forward(self, decoder_input_ids, encoder_output, encoder_padding_mask):
        """
        decoder_input_ids: (B, T_dec)  - Decoder input tokens
        encoder_output: (B, T_enc, embed_dim) - Encoder's output
        encoder_padding_mask: (B, 1, T_enc) - Encoder's padding mask, used for cross-attention
        """
        B, T_dec = decoder_input_ids.shape
        device = decoder_input_ids.device
        
        # ========== MASK CREATION ==========
        
        # 1. Causal Mask
        #    Prevents "looking ahead" to future tokens
        #    Shape: (1, T_dec, T_dec)
        causal_mask = self._create_causal_mask(T_dec, device)
        
        # 2. Decoder Padding Mask
        #    Prevents attention on [PAD] tokens
        #    Shape: (B, 1, T_dec)
        decoder_padding_mask = self._create_padding_mask(decoder_input_ids, self.pad_token_id)

        # 3. **Key: Combine Masks**
        #    Self-attention needs to satisfy *both* conditions:
        #    a. Cannot look at [PAD] (from decoder_padding_mask)
        #    b. Cannot look at "future" (from causal_mask)
        #    We use '&' (logical AND) to combine them
        #    Broadcast: (B, 1, T_dec) & (1, T_dec, T_dec) -> (B, T_dec, T_dec)
        self_attn_mask = decoder_padding_mask & causal_mask
        
        # 4. Cross-Attention Mask
        #    This is simple, it's just the encoder_padding_mask passed from the Encoder
        #    It tells the Decoder not to pay attention to the [PAD] sections in the Encoder output
        #    Shape: (B, 1, T_enc)
        cross_attn_mask = encoder_padding_mask
        
        # ========== END MASK CREATION ==========
        
        # 1. Calculate Embeddings
        positions = torch.arange(T_dec, device=device).unsqueeze(0) # (1, T_dec)
        tok_emb = self.token_embedding(decoder_input_ids) * self.scale
        pos_emb = self.pos_embedding(positions)
        x = self.dropout(tok_emb + pos_emb) # (B, T_dec, embed_dim)

        # 2. Pass through Decoder Blocks layer by layer
        for layer in self.layers:
            x = layer(
                x, 
                encoder_output,     # K, V
                self_attn_mask,     # **Mask 1** passed to self-attention
                cross_attn_mask     # **Mask 2** passed to cross-attention
            )
        
        # 3. Final Output
        logits = self.final_linear(x) # (B, T_dec, vocab_size)
        return logits

if __name__ == "__main__":
    # Settings
    batch_size = 2
    max_seq_len = 10
    src_len = 5 # Encoder output length
    tgt_len = 6 # Decoder input length
    embed_dim = 16
    num_heads = 4
    ffn_dim = 32
    vocab_size = 100
    num_layers = 2
    pad_token_id = 0
    
    print(f"Configuration: batch_size={batch_size}, src_len={src_len}, tgt_len={tgt_len}, embed_dim={embed_dim}")

    # Instantiate the Decoder
    decoder = Decoder(
        num_layers=num_layers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        pad_token_id=pad_token_id
    )
    
    # Create dummy inputs
    
    # 1. Decoder Input IDs (B, T_dec)
    #    Let's simulate some padding at the end of the second sequence
    decoder_input_ids = torch.randint(1, vocab_size, (batch_size, tgt_len))
    decoder_input_ids[1, -2:] = pad_token_id # Pad the last 2 tokens of the 2nd batch
    
    # 2. Encoder Output (B, T_enc, embed_dim)
    #    This would come from the Encoder in a real Transformer
    encoder_output = torch.randn(batch_size, src_len, embed_dim)
    
    # 3. Encoder Padding Mask (B, 1, T_enc)
    #    Simulate padding in the encoder source
    encoder_padding_mask = torch.ones(batch_size, 1, src_len)
    encoder_padding_mask[1, 0, -1] = 0 # Mask the last token of the 2nd batch in encoder
    
    print("\n--- Input Shapes ---")
    print(f"Decoder Input: {decoder_input_ids.shape}")
    print(f"Encoder Output: {encoder_output.shape}")
    print(f"Encoder Mask: {encoder_padding_mask.shape}")

    # Forward pass
    logits = decoder(decoder_input_ids, encoder_output, encoder_padding_mask)
    
    print("\n--- Output ---")
    print(f"Logits shape: {logits.shape}")
    
    # Verification
    expected_shape = (batch_size, tgt_len, vocab_size)
    if logits.shape == expected_shape:
        print("Test Passed: Logits shape is correct.")
    else:
        print(f"Test Failed: Expected {expected_shape}, got {logits.shape}")
