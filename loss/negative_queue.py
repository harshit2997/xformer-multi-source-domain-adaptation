import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NegUniform(nn.Module):
    def __init__(self, k, temperature=0.01):
        super(NegUniform, self).__init__()
        self.k = k
        self.temperature = temperature
        self.v = 0.95
        self.decay = F.normalize(torch.tensor([self.v**i for i in range(k)]), p=1, dim=0).unsqueeze(dim=0).cuda()

    @staticmethod
    def cosine_sim(a:torch.Tensor, b: torch.Tensor):
        a = F.normalize(a)
        b = F.normalize(b)
        a = torch.unsqueeze(a, dim=1)
        b = torch.unsqueeze(b, dim=0)
        sim_mat = torch.sum(a * b, dim=2)
        return sim_mat

    def forward(self, feature, target, negative_features, idx):
        sims = []
        for target_feature in negative_features:
            sim = self.cosine_sim(feature, target_feature)
            sims.append(sim)

        n = feature.shape[0]
        mask = target.expand(n, n).eq(target.expand(n, n).t()).float()

        neg_top = []
        for i in range(n):
            tmp = []
            for j, sim in enumerate(sims):
                if j == idx:
                    top_val, _ = torch.topk(sim[i][mask[i] == 0], self.k)
                    tmp.append(top_val)
                else:
                    top_val, _ = torch.topk(sim[i], self.k)
                    tmp.append(top_val)
            tmp = torch.stack(tmp, dim=0)
            neg_top.append(tmp)
        neg_top = torch.stack(neg_top, dim=0)
        neg_top = neg_top / self.temperature
        neg_top = torch.softmax(neg_top, dim=1)
        loss = torch.mean(torch.sum(torch.sum(neg_top*torch.log(neg_top), dim=1)*self.decay, dim=1)) + math.log(len(negative_features))
        return loss


if __name__ == '__main__':
    feature = torch.randn((4, 256))
    targets = torch.arange(4)
    negative_features = [torch.randn(4, 256), torch.randn(4, 256), torch.randn(4, 256)]
    loss = NegUniform(2)
    t = loss(feature, targets, negative_features, 1)
