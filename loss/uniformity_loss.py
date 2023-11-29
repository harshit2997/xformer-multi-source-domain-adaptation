import torch
import torch.nn as nn
import torch.nn.functional as F


class Uniformity(nn.Module):
    def __init__(self, q_size=4096, unif_t=2, num_queues=3):
        super(Uniformity, self).__init__()
        self.qs = [Queue(q_size) for i in range(num_queues)]
        self.num_queues = num_queues
        self.unif_t = unif_t

    @staticmethod
    def cosine_sim(a: torch.Tensor, b: torch.Tensor):
        a = F.normalize(a)
        b = F.normalize(b)
        sim_mat = torch.matmul(a, b.t())
        return sim_mat

    def deque_enqueue(self, ema_features, ema_targets):
        for idx, (ema_feature, ema_target) in enumerate(list(zip(ema_features, ema_targets))):
            self.qs[idx].deque_enqueue(ema_feature.detach(), ema_target.detach())

    def forward(self, x, targets, order):
        distmats = []

        for q in self.qs:
            ema_feature, ema_target = q.get_current_queue()
            sim = self.cosine_sim(x, ema_feature)
            distmats.append(2 - 2 * sim)

        _, cur_ema_target = self.qs[order].get_current_queue()
        mask = targets.unsqueeze(dim=1).eq(cur_ema_target.unsqueeze(dim=1).t())
        n = x.shape[0]

        loss = 0
        for i in range(n):
            tmp_dist = []
            for idx in range(self.num_queues):
                dist = distmats[idx][i]
                if idx == order:
                    tmp = dist[mask[i] == 0]
                    tmp_dist.append(tmp)
                else:
                    tmp_dist.append(dist)
            tmp_dist = torch.cat(tmp_dist, dim=0)
            loss += torch.log(torch.mean(torch.exp(-self.unif_t*tmp_dist)))
        loss = loss / n

        return loss


class Queue(object):
    def __init__(self, q_size=4096, feat_dim=512):
        self.q_size = q_size
        self.feat_dims = feat_dim
        self.ptr = 0

        self.data = torch.zeros((q_size, feat_dim)).cuda()
        self.targets = torch.zeros(q_size, dtype=torch.long).cuda()

    def is_empty(self):
        return self.ptr == 0

    def is_full(self):
        return self.ptr == self.q_size

    def deque_enqueue(self, x, targets):
        batch_size = x.shape[0]
        if self.q_size >= batch_size:
            assert self.q_size % batch_size == 0
        else:
            x = x[:self.q_size]
            targets = targets[:self.q_size]
            batch_size = x.shape[0]

        if self.ptr == self.q_size:
            self.data = torch.cat([self.data[batch_size:], x], dim=0)
            self.targets = torch.cat([self.targets[batch_size:], targets], dim=0)
        else:
            self.data[self.ptr:self.ptr+batch_size] = x
            self.targets[self.ptr:self.ptr + batch_size] = targets
            self.ptr += batch_size

    def get_current_queue(self):
        return self.data[:self.ptr], self.targets[:self.ptr]






