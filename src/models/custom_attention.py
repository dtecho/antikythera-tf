import torch.nn as nn

class CustomAttention(nn.Module):
    def __init__(self, num_heads, gear_ratios):
        super().__init__()
        self.num_heads = num_heads
        self.gear_ratios = gear_ratios
        self.attn = nn.MultiheadAttention(embed_dim=num_heads, num_heads=num_heads)

    def forward(self, query, key, value):
        # Scale attention weights by gear ratios
        scaling = torch.tensor(self.gear_ratios).unsqueeze(0).unsqueeze(2)
        attn_output, _ = self.attn(query * scaling, key * scaling, value * scaling)
        return attn_output
