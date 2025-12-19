import torch
import torch.nn as nn

try:
    from .SingleHeadAttention import SingleHeadAttention
except ImportError:  # pragma: no cover
    from SingleHeadAttention import SingleHeadAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, f"blah blah blah"
        head_dim = embed_dim // num_heads

        self.heads = nn.ModuleList([
            SingleHeadAttention(embed_dim, head_dim) for _ in range(num_heads)
        ])
        
        # concat the final linear layer
        self.out_proj = nn.Linear(embed_dim, embed_dim) # (num_heads * head_dim) -> embed_dim
    
    def forward(self, q_input, k_input, v_input, mask=None):
        head_outputs = []
        for head in self.heads:
            head_outputs.append(head(q_input, k_input, v_input, mask))
        context = torch.cat(head_outputs, dim=-1) # (B, T_q, embed_dim)
        
        return self.out_proj(context)

if __name__ == "__main__":
    # Settings
    batch_size = 2
    seq_len = 5
    embed_dim = 16  # Must be divisible by num_heads
    num_heads = 4
    
    print(f"Configuration: batch_size={batch_size}, seq_len={seq_len}, embed_dim={embed_dim}, num_heads={num_heads}")

    # Instantiate the module
    mha = MultiHeadAttention(embed_dim, num_heads)
    
    # Create dummy inputs
    q_input = torch.randn(batch_size, seq_len, embed_dim)
    k_input = torch.randn(batch_size, seq_len, embed_dim)
    v_input = torch.randn(batch_size, seq_len, embed_dim)
    
    # Test 1: Forward pass without mask
    output = mha(q_input, k_input, v_input)
    print("\n--- No Mask Test ---")
    print(f"Input shape: {q_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Verify shape: should be (B, T, embed_dim)
    expected_shape = (batch_size, seq_len, embed_dim)
    if output.shape == expected_shape:
        print("Test Passed: Output shape is correct.")
    else:
        print(f"Test Failed: Expected {expected_shape}, got {output.shape}")

    # Test 2: Forward pass with mask
    print("\n--- Masked Test ---")
    mask = torch.tril(torch.ones(seq_len, seq_len))
    output_masked = mha(q_input, k_input, v_input, mask=mask)
    print(f"Mask shape: {mask.shape}")
    print(f"Output shape: {output_masked.shape}")
    
    if output_masked.shape == expected_shape:
        print("Test Passed: Output shape with mask is correct.")
    else:
        print(f"Test Failed: Expected {expected_shape}, got {output_masked.shape}")
