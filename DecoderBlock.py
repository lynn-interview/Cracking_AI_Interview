import torch
import torch.nn as nn

try:
    from .MultiHeadAttention import MultiHeadAttention
except ImportError:  # pragma: no cover
    from MultiHeadAttention import MultiHeadAttention

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        # 1. masked self attention
        self.masked_self_attn = MultiHeadAttention(embed_dim, num_heads)
        
        # 2. cross-attention
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads)
        
        # 3. FFN
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, self_attn_mask, cross_attn_mask):
        """
        x:                 (B, T_dec, embed_dim)
        encoder_output:    (B, T_enc, embed_dim)
        self_attn_mask:    (B, T_dec, T_dec)
        cross_attn_mask:   (B, 1, T_enc)
        """
        
        # Masked Self-Attention
        # Q, K, V are from x
        attn_output = self.masked_self_attn(
            q_input=x, 
            k_input=x, 
            v_input=x, 
            mask=self_attn_mask
        )
        x = self.norm1(x + self.dropout(attn_output)) # Add & Norm
        
        # Cross-Attention
        # Q is from x (Decoder), K and V are from encoder_output (Encoder)
        attn_output = self.cross_attn(
            q_input=x,              
            k_input=encoder_output, 
            v_input=encoder_output, 
            mask=cross_attn_mask
        )
        x = self.norm2(x + self.dropout(attn_output)) # Add & Norm
        
        #FFN
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output)) # Add & Norm
        
        return x

if __name__ == "__main__":
    # Settings
    batch_size = 2
    seq_len_dec = 5
    seq_len_enc = 7
    embed_dim = 16
    num_heads = 4
    ffn_dim = 32
    
    print(f"Configuration: batch_size={batch_size}, seq_len_dec={seq_len_dec}, seq_len_enc={seq_len_enc}, embed_dim={embed_dim}")

    # Instantiate the block
    decoder_block = DecoderBlock(embed_dim, num_heads, ffn_dim)
    
    # Create dummy inputs
    x = torch.randn(batch_size, seq_len_dec, embed_dim) # Decoder input
    encoder_output = torch.randn(batch_size, seq_len_enc, embed_dim) # Encoder output
    
    # Create dummy masks
    # Self-attention mask (causal): (B, T_dec, T_dec)
    # Using triangular mask for causality
    causal_mask = torch.tril(torch.ones(seq_len_dec, seq_len_dec))
    # Add batch dim for broadcasting
    self_attn_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
    
    # Cross-attention mask: (B, 1, T_enc)
    # Let's say we mask the last token of the encoder output
    cross_attn_mask = torch.ones(batch_size, 1, seq_len_enc)
    cross_attn_mask[:, :, -1] = 0 # Mask last token
    
    print("\n--- Input Shapes ---")
    print(f"x (Decoder Input): {x.shape}")
    print(f"encoder_output: {encoder_output.shape}")
    print(f"self_attn_mask: {self_attn_mask.shape}")
    print(f"cross_attn_mask: {cross_attn_mask.shape}")

    # Forward pass
    output = decoder_block(x, encoder_output, self_attn_mask, cross_attn_mask)
    
    print("\n--- Output ---")
    print(f"Output shape: {output.shape}")
    
    expected_shape = (batch_size, seq_len_dec, embed_dim)
    if output.shape == expected_shape:
        print("Test Passed: Output shape is correct.")
    else:
        print(f"Test Failed: Expected {expected_shape}, got {output.shape}")
