from __future__ import print_function, absolute_import

import time
from collections import OrderedDict

import numpy as np
import torch

from reid.utils import to_torch, to_numpy
from reid.utils.meters import AverageMeter


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, verbose=True):
        features, _ = self.extract_features(self.model, data_loader, verbose=verbose)

        distmat = self.pairwise_distance(features, query, gallery)

        return self.eval(distmat, query, gallery)

    def eval(self, distmat, query, gallery):
        distmat = to_numpy(distmat)

        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]

        query_ids = np.asarray(query_ids)
        gallery_ids = np.asarray(gallery_ids)
        query_cams = np.asarray(query_cams)
        gallery_cams = np.asarray(gallery_cams)

        cmc_scores, mAP = self.eval_func(distmat, q_pids=query_ids, g_pids=gallery_ids,
                                    q_camids=query_cams, g_camids=gallery_cams, max_rank=50)

        print("=" * 80)
        print('Mean AP: {:4.1%}'.format(mAP))
        print('CMC Scores:')
        cmc_topk = (1, 5, 10, 20, 50)
        for k in cmc_topk:
            print('  top-{:<4}{:12.1%}'.format(k, cmc_scores[k - 1]))
        return mAP, cmc_scores[0]

    @staticmethod
    def pairwise_distance(features, query=None, gallery=None):
        if query is None and gallery is None:
            n = len(features)
            x = torch.cat(list(features.values()))
            x = x.view(n, -1)
            dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
            dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
            return dist_m

        x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
        y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
        m, n = x.size(0), y.size(0)
        x = x.view(m, -1)
        y = y.view(n, -1)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                 torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        dist_m.addmm_(1, -2, x, y.t())
        return dist_m

    @staticmethod
    def extract_features(model, data_loader, print_freq=50, verbose=True):
        model.eval()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        features = OrderedDict()
        labels = OrderedDict()

        end = time.time()
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                imgs = data[0]
                fnames = data[1]
                pids = data[2]

                data_time.update(time.time() - end)

                inputs = to_torch(imgs).cuda()
                outputs = model(inputs)
                outputs = outputs.data.cpu()

                for fname, output, pid in zip(fnames, outputs, pids):
                    features[fname] = output
                    labels[fname] = pid

                batch_time.update(time.time() - end)
                end = time.time()

                if (i + 1) % print_freq == 0 and verbose:
                    print('Extract Features: [{}/{}]\t'
                          'Time {:.3f} ({:.3f})\t'
                          'Data {:.3f} ({:.3f})\t'
                          .format(i + 1, len(data_loader),
                                  batch_time.val, batch_time.avg,
                                  data_time.val, data_time.avg))

        return features, labels

    @staticmethod
    def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50, ap_topk=None):
        """Evaluation with market1501 metric
            Key: for each query identity, its gallery images from the same camera view are discarded.
            """
        if (ap_topk is not None):
            assert (ap_topk >= max_rank)
        num_q, num_g = distmat.shape
        if num_g < max_rank:
            max_rank = num_g
            print("Note: number of gallery samples is quite small, got {}".format(num_g))
        indices = np.argsort(distmat, axis=1)
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

        # compute cmc curve for each query
        all_cmc = []
        all_AP = []
        num_valid_q = 0.  # number of valid query
        for q_idx in range(num_q):
            # get query pid and camid
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]

            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)

            # compute cmc curve
            # binary vector, positions with value 1 are correct matches
            # orig_cmc = matches[q_idx][keep]

            if not np.any(matches[q_idx][keep]):
                # if not np.any(orig_cmc):
                # this condition is true when query identity does not appear in gallery
                continue

            if (ap_topk is None):
                orig_cmc = matches[q_idx][keep]
            else:
                orig_cmc = matches[q_idx][:ap_topk][keep[:ap_topk]]
                # orig_cmc = matches[q_idx][keep][:ap_topk]

            cmc = orig_cmc.cumsum()
            cmc[cmc > 1] = 1

            all_cmc.append(cmc[:max_rank])
            num_valid_q += 1.

            # compute average precision
            # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
            num_rel = orig_cmc.sum()
            tmp_cmc = orig_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / max(1, num_rel)
            all_AP.append(AP)

        assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)

        return all_cmc, mAP

