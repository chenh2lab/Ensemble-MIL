import torch 
import torch.nn as nn
import torch.nn.functional as F

# attention model
class AttnClassifier(nn.Module):
    def __init__(self, input_dim, attn_dim, output_dim, dropout = True, p_dropout = 0.5):
        super(AttnClassifier, self).__init__()

        att_v = [nn.Linear(input_dim, attn_dim), nn.Tanh()]
        att_u = [nn.Linear(input_dim, attn_dim), nn.Sigmoid()]

        if dropout:
            att_v.append(nn.Dropout(p_dropout))
            att_u.append(nn.Dropout(p_dropout))

        self.attention_V = nn.Sequential(*att_v)
        self.attention_U = nn.Sequential(*att_u)
        self.attention_weights = nn.Linear(attn_dim, 1)

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # batch size must be 1 !!!
        x = x.squeeze(0)   # (N, input_dim)

        alpha_V = self.attention_V(x)   # (N, 1042)
        alpha_U = self.attention_U(x)   # (N, 1024)
        alpha = self.attention_weights(alpha_V * alpha_U)   # element wise muliplication (N, 1)
        alpha = torch.transpose(alpha, 1, 0) # (1, N)
        alpha = F.softmax(alpha, dim = 1)

        M = torch.mm(alpha, x)   # (1, input_dim)
        out = self.fc(M) # (1, 2)
        return out, alpha
