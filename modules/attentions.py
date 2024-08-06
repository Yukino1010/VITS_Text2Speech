import torch
from torch import nn
from modules import LayerNorm
   
    
class FFN(nn.Module):
    def __init__(self, in_channels, out_channels, filter_channels,\
                  kernel_size, p_dropout=0., activation='relu'):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.drop = nn.Dropout(p_dropout)
        self.activation = activation

    def forward(self, x, x_mask):
        x = self.conv1(x * x_mask)
        if self.activation == "gelu":
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)
        x = self.drop(x)
        x = self.conv2(x * x_mask)
        return x * x_mask
    
class Encoder(nn.Module):
    def __init__(self, hidden_channels, filter_channels, n_heads, \
                 n_layers, kernel_size=1, p_dropout=0., window_size=4):
        
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        self.drop = nn.Dropout(p_dropout)
        self.attn= nn.ModuleList()
        self.norm1 = nn.ModuleList()
        self.ffn = nn.ModuleList()
        self.norm2 = nn.ModuleList()

        for _ in range(self.n_layers):
            self.attn.append(nn.MultiheadAttention(hidden_channels, n_heads, dropout=p_dropout))
            self.norm1.append(LayerNorm(hidden_channels))
            self.ffn.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout))
            self.norm2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask):
        batch_size, _, seq_len = x_mask.shape
        attn_mask = x_mask.expand(batch_size, seq_len, seq_len)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) 
        attn_mask = attn_mask.reshape(batch_size * self.n_heads, seq_len, seq_len)

        for i in range(self.n_layers):
            x_t = x.permute(2, 0, 1)  # (seq_len, batch_size, embed_dim)
            y, _ = self.attn[i](x_t, x_t, x_t, attn_mask=attn_mask)
            y = y.permute(1, 2, 0)  # (batch_size, embed_dim, seq_len)
            y = self.drop(y)
            x = self.norm1[i](x + y)

            y = self.ffn[i](x, x_mask)
            y = self.drop(y)
            x = self.norm2[i](x + y)

        x = x * x_mask
        return x
