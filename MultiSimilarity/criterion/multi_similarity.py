import torch
import torch.nn as nn


class MultiSimilarityLoss(nn.Module):
    def __init__(self, alpha, beta, lamda, epsilon):
        super(MultiSimilarityLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.lamda = lamda
        self.epsilon = epsilon
    
    def forward(self, x, y):
        S = x @ x.t()

        loss = []
        for i in range(x.shape[0]):
            pos_idx = y == y[i]
            pos_idx[i] = 0
            neg_idx = y != y[i]

            S_pos = S[i][pos_idx]
            S_neg = S[i][neg_idx]

            neg_idx = (S_neg + self.epsilon) > torch.min(S_pos)
            pos_idx = (S_pos - self.epsilon) < torch.max(S_neg)

            if not torch.sum(neg_idx) or not torch.sum(pos_idx):
                continue

            S_pos = S_pos[pos_idx]
            S_neg = S_neg[neg_idx]

            pos_loss = 1. / self.alpha * torch.log(1 + torch.sum(torch.exp(-self.alpha * (S_pos - self.lamda))))
            neg_loss = 1. / self.beta * torch.log(1 + torch.sum(torch.exp(self.beta * (S_neg - self.lamda))))
            loss.append(pos_loss + neg_loss)
        if len(loss) == 0:
            return 0 * S.mean()
        else:
            return torch.mean(torch.stack(loss))
