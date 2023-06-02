""" loss for FedETuning """

from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.autograd import Variable

from utils.register import registry


class BaseLoss(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.logger = registry.get("logger")

    def forward(self, x, target):
        raise NotImplementedError


@registry.register_loss("focal")
class FocalLoss(BaseLoss):
    """FocalLoss."""

    def __init__(self, config):
        super().__init__(config)
        self.gamma = config.gamma
        self.alpha = config.alpha
        if isinstance(self.alpha, (float, int)):
            self.alpha = torch.Tensor([self.alpha, 1 - self.alpha])
        if isinstance(self.alpha, list):
            self.alpha = torch.Tensor(self.alpha)
        self.size_average = config.size_average

    def forward(self, x, target):
        target = target.squeeze().long()
        if x.dim() > 2:
            x = x.view(x.size(0), x.size(1), -1)  # N,C,H,W => N,C,H*W
            x = x.transpose(1, 2)  # N,C,H*W => N,H*W,C
            x = x.contiguous().view(-1, x.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(x)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != x.data.type():
                self.alpha = self.alpha.type_as(x.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


@registry.register_loss("label_smooth")
class LabelSmoothLoss(BaseLoss):
    """Label Smoothing FocalLoss."""

    def __init__(self, config):
        super().__init__(config)
        self.gamma = config.get("gamma", 2)
        self.smooth_eps = config.get("smooth_eps", 0.25)
        self.size_average = config.get("size_average", True)

    def forward(self, x, target):
        cls_num = x.size(1)
        target = target.squeeze().long()
        if x.dim() > 2:
            x = x.view(x.size(0), x.size(1), -1)  # N,C,H,W => N,C,H*W
            x = x.transpose(1, 2)  # N,C,H*W => N,H*W,C
            x = x.contiguous().view(-1, x.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        pred = F.log_softmax(x)
        logpt = pred.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        focal_loss = -1 * (1 - pt) ** self.gamma * logpt

        lce = -1 * torch.sum(pred, dim=1) / cls_num
        loss = (1 - self.smooth_eps) * focal_loss + self.smooth_eps * lce

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


@registry.register_loss("kl")
class DistillationLoss(BaseLoss):
    """DistillationLoss."""

    def __init__(self, config):
        super().__init__(config)
        self.temp = self.config.temp
        # self.reduction_kd = self.configs.get("reduction_kd", "batchmean")
        self.criterion = nn.KLDivLoss(reduction="batchmean")

    def forward(self, x, y):
        student = F.log_softmax(x / self.temp, dim=1)
        teacher = F.softmax(y / self.temp, dim=1)
        klloss = self.criterion(student, teacher) * self.temp * self.temp
        return klloss


def normalize(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """Normalizing to unit length along the specified dimension.

    Args:
        x (torch.Tensor): x
        axis (int): axis

    Returns:
        torch.Tensor:
    """
    x = 1.0 * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """euclidean_dist.

    Args:
        x (torch.Tensor): in shape of [m, d]
        y (torch.Tensor): in shape of [n, d]

    Returns:
        torch.Tensor: in shape of [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True
    )
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True
    )
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (
            labels.new()
            .resize_as_(labels)
            .copy_(torch.arange(0, N).long())
            .unsqueeze(0)
            .expand(N, N)
        )
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data
        )
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data
        )
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an  # , p_inds, n_inds

    return dist_ap, dist_an


@registry.register_loss("xent")
class CrossEntropyLoss(BaseLoss):
    def __init__(self, config):
        super().__init__(config)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, target):
        target = target.squeeze().long()
        return self.loss_fn(x, target)


@registry.register_loss("bce")
class BCELoss(BaseLoss):
    def __init__(self, config):
        super().__init__(config)
        self.loss_fn = nn.BCELoss()

    def forward(self, x, target):
        return self.loss_fn(x, target)


@registry.register_loss("bcewlogits")
class BCEWithLogitsLoss(BaseLoss):
    def __init__(self, config):
        super().__init__(config)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x, target):
        return self.loss_fn(x, target)


@registry.register_loss("l1")
class L1Loss(BaseLoss):
    def __init__(self, config):
        super().__init__(config)
        self.loss_fn = nn.L1Loss()

    def forward(self, x, target):
        return self.loss_fn(x, target)


@registry.register_loss("mse")
class MSELoss(BaseLoss):
    def __init__(self, config):
        super().__init__(config)
        self.loss_fn = nn.MSELoss()

    def forward(self, x, target):
        return self.loss_fn(x, target)


@registry.register_loss("smoothl1")
class SmoothL1Loss(BaseLoss):
    def __init__(self, config):
        super().__init__(config)
        self.loss_fn = nn.SmoothL1Loss()

    def forward(self, x, target):
        return self.loss_fn(x, target)


class Loss(nn.Module):

    def __init__(self):
        super().__init__()

        self.config = registry.get("loss_config")

        self._build()

    def _build(self):
        logger.debug("create criterion")
        self.losses = registry.get_loss_class(self.config.type)(self.config)

    def forward(self, x, target: torch.Tensor):
        return self.losses(x, target)
