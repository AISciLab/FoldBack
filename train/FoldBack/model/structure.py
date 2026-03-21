import torch
import torch.nn as nn
import torch.nn.functional as F

class Seq3DAttention(nn.Module):
    def __init__(self, struct_dim=384,seq_dim=1152):
        super(Seq3DAttention, self).__init__()
        self.struct_proj1 = nn.Linear(struct_dim, 320,bias=False).float()
        self.struct_proj2 = nn.Linear(seq_dim, 320,bias=False).float()
        self.struct_q_proj = nn.Linear(320, 320,bias=False).float()
        self.struct_k_proj = nn.Linear(320, 320,bias=False).float()
        self.struct_v_proj = nn.Linear(320, 320,bias=False).float()
        self.struct_out = nn.Linear(320, 320,bias=False).float()
    def forward(self, seq_emb,struct_emb,struct_mask):

        struct_emb = self.struct_proj1(struct_emb)
        seq_emb = self.struct_proj2(seq_emb)
        fussion_emb = torch.cat((seq_emb,struct_emb),dim=1)
        Q = self.struct_q_proj(fussion_emb)
        K = self.struct_k_proj(fussion_emb)
        V = self.struct_v_proj(fussion_emb)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))
        dim = Q.size(-1)
        attn_scores = attn_scores / torch.sqrt(torch.tensor(dim).float())

        if struct_mask is not None:
            struct_mask = struct_mask.unsqueeze(1)  # (batch_size, 1, seq_len)
            attn_scores = attn_scores.masked_fill(struct_mask == 0, float('-inf'))
        attention_weights = F.softmax(attn_scores, dim=-1)

        out = torch.matmul(attention_weights, V)
        out = self.struct_out(out)
        out = out + fussion_emb
        return out


