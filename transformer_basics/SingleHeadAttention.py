import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SingleHeadAttention(nn.Module):
    """
    still the same logics as discussed before，but the inputs become generalized
    - previously: q_input, k_input, v_input are wrapped up to a single 'x'
    - cross attention: q_input becomes 'x', k_input/v_input becomes 'encoder_output'
    """
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.head_dim = head_dim

        self.q_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, head_dim, bias=False)
        
    def forward(self, q_input, k_input, v_input, mask=None):
        q = self.q_proj(q_input) # (B, T_q, head_dim)
        k = self.k_proj(k_input) # (B, T_kv, head_dim)
        v = self.v_proj(v_input) # (B, T_kv, head_dim)
        
        scale = 1 / math.sqrt(self.head_dim)
        attn_scores = (q @ k.transpose(-2, -1)) * scale # (B, T_q, T_kv)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        return attn_weights @ v # (B, T_q, head_dim)

if __name__ == "__main__":
    # Settings
    batch_size = 2
    seq_len = 3
    embed_dim = 8
    head_dim = 4
    
    print(f"Configuration: batch_size={batch_size}, seq_len={seq_len}, embed_dim={embed_dim}, head_dim={head_dim}")

    # Instantiate the module
    sha = SingleHeadAttention(embed_dim, head_dim)
    
    # Create dummy inputs (Batch, Seq_Len, Embed_Dim)
    q_input = torch.randn(batch_size, seq_len, embed_dim)
    k_input = torch.randn(batch_size, seq_len, embed_dim)
    v_input = torch.randn(batch_size, seq_len, embed_dim)
    
    # Forward pass without mask
    output = sha(q_input, k_input, v_input)
    print("\n--- No Mask Test ---")
    print(f"Input shape: {q_input.shape}")
    print(f"Output shape: {output.shape}")
    
    expected_shape = (batch_size, seq_len, head_dim)
    if output.shape == expected_shape:
        print("Test Passed: Output shape is correct.")
    else:
        print(f"Test Failed: Expected {expected_shape}, got {output.shape}")

    # Forward pass with causal mask
    print("\n--- Masked Test (Causal) ---")
    # Lower triangular mask for autoregressive property
    mask = torch.tril(torch.ones(seq_len, seq_len))
    output_masked = sha(q_input, k_input, v_input, mask=mask)
    print(f"Mask shape: {mask.shape}")
    print(f"Output shape: {output_masked.shape}")
    
    if output_masked.shape == expected_shape:
        print("Test Passed: Output shape with mask is correct.")
    else:
        print(f"Test Failed: Expected {expected_shape}, got {output_masked.shape}")
