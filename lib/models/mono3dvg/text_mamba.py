import torch
import torch.nn.functional as F
from torch import nn


class TextMambaBlock(nn.Module):
    def __init__(self, d_model, expand_ratio=2, kernel_size=3, dropout=0.1):
        super().__init__()
        inner_dim = max(d_model, int(d_model * expand_ratio))

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, inner_dim * 2)
        self.depthwise_conv = nn.Conv1d(
            inner_dim,
            inner_dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=inner_dim,
        )
        self.delta_proj = nn.Linear(inner_dim, inner_dim)
        self.out_proj = nn.Linear(inner_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def _selective_scan(self, values, deltas, key_padding_mask=None):
        batch_size, seq_len, hidden_dim = values.shape
        state = values.new_zeros(batch_size, hidden_dim)
        outputs = []

        for step in range(seq_len):
            new_state = deltas[:, step] * state + (1.0 - deltas[:, step]) * values[:, step]
            if key_padding_mask is not None:
                valid = (~key_padding_mask[:, step]).unsqueeze(-1).type_as(values)
                state = valid * new_state + (1.0 - valid) * state
                outputs.append(valid * state)
            else:
                state = new_state
                outputs.append(state)

        return torch.stack(outputs, dim=1)

    def forward(self, x, key_padding_mask=None):
        residual = x
        x = self.norm(x)
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        values, gate = self.in_proj(x).chunk(2, dim=-1)
        values = self.depthwise_conv(values.transpose(1, 2))[..., :x.shape[1]].transpose(1, 2)
        values = F.silu(values)
        deltas = torch.sigmoid(self.delta_proj(values))

        forward_state = self._selective_scan(values, deltas, key_padding_mask)

        reverse_mask = torch.flip(key_padding_mask, dims=[1]) if key_padding_mask is not None else None
        backward_state = self._selective_scan(
            torch.flip(values, dims=[1]),
            torch.flip(deltas, dims=[1]),
            reverse_mask,
        )
        backward_state = torch.flip(backward_state, dims=[1])

        fused = (forward_state + backward_state) * torch.sigmoid(gate)
        fused = self.out_proj(self.dropout(fused))
        output = residual + fused
        if key_padding_mask is not None:
            output = output.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return output


class TextMambaEncoder(nn.Module):
    def __init__(self, d_model, num_layers=1, expand_ratio=2, kernel_size=3, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TextMambaBlock(
                d_model=d_model,
                expand_ratio=expand_ratio,
                kernel_size=kernel_size,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        return x
