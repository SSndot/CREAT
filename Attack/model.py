import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionEncoder(nn.Module):
    def __init__(self, max_seq_len: int, pos_emb_size: int):
        super(PositionEncoder, self).__init__()
        self.pos_emb_size = pos_emb_size
        self.max_seq_len = max_seq_len
        self.pos_emb = nn.Embedding(max_seq_len, pos_emb_size)
        nn.init.xavier_uniform_(self.pos_emb.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(self.max_seq_len).unsqueeze(0).expand(x.size(0), -1).to(x.device)
        return self.pos_emb(pos)


class GeneratorModel(nn.Module):
    def __init__(self,
                 seq_emb_size: int,
                 item_emb_size: int,
                 hidden_size: int,
                 max_seq_len: int,
                 pos_emb_size: int = 64,
                 dropout_rate: float = 0.1,
                 num_layers: int = 2,
                 device: str = 'cpu'):
        super(GeneratorModel, self).__init__()
        self.num_layers = num_layers
        inp_size = seq_emb_size + item_emb_size + pos_emb_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(inp_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)
        self.pos_embedding = PositionEncoder(max_seq_len, pos_emb_size)

        self.device = device

    def forward(self, seq_emb: torch.Tensor, item_emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, item_emb_dim = item_emb.shape
        seq_emb = seq_emb.unsqueeze(1).expand(-1, seq_len, -1)
        pos_emb = self.pos_embedding(seq_emb)
        inp_emb = torch.cat([seq_emb, item_emb, pos_emb], dim=2)    # batch_size x seq_len x (seq_emb + item_emb + pos_emb)

        hidden = torch.zeros(self.num_layers, seq_emb.size(0), self.hidden_size).to(self.device)
        out, _ = self.gru(inp_emb, hidden)
        out = self.dropout(out)
        out = self.fc(out)
        out = out.squeeze(-1)
        out = out.masked_fill(mask == 0, float('-inf'))
        out = F.softmax(out, dim=-1)
        return out
