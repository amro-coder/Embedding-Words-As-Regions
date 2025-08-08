import torch 
import torch.nn.functional as F


def max_margin_loss(positive_score, negative_score, margin): 
    loss_per_batch = torch.relu((negative_score - positive_score) + margin).mean(dim = 1)
    return loss_per_batch.mean(), loss_per_batch.shape[0] - loss_per_batch.count_nonzero().item(), loss_per_batch


def negative_sampling_loss(positive_score, negative_score):
    loss_per_batch = - F.logsigmoid(positive_score).squeeze(-1) - F.logsigmoid(-negative_score).sum(dim=1)
    return loss_per_batch.mean(), 0, loss_per_batch


