import torch
from torch.nn import functional
from torch import nn
import numpy as np

class TemperatureAnneal:
    def __init__(self, initial_temp=1.0, anneal_rate=0.0, min_temp=0.5, device=torch.device('cuda')):
        self.initial_temp = initial_temp
        self.anneal_rate = anneal_rate
        self.min_temp = min_temp
        self.device = device

        self._temperature = self.initial_temp
        self.last_epoch = 0

    def get_temp(self):
        return self._temperature

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        current_temp = self.initial_temp * np.exp(-self.anneal_rate * self.last_epoch)
        # noinspection PyArgumentList
        self._temperature = torch.max(torch.FloatTensor([current_temp, self.min_temp]).to(self.device))

    def reset(self):
        self._temperature = self.initial_temp
        self.last_epoch = 0

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


def softmax(logits, temperature=1.0, dim=1):
    exps = torch.exp(logits/temperature)
    return exps/torch.sum(exps, dim=dim)


def gumbel(size, device=torch.device('cuda:1'), eps=1e-8):
    return -torch.log(-torch.log(torch.rand(size, device=device) + eps) + eps)


def sample_gumbel_softmax(logits, temperature=1.0, dim=1, device=torch.device('cuda:1')):
    return functional.softmax((logits.to(device) + gumbel(logits.size(),device).to(device))/temperature, dim=dim)


def gumbel_softmax(logits, temperature=1.0, dim=1, hard=True, is_prob=True, device=torch.device('cuda:1')):
    if is_prob:
        logits = torch.log(logits + 1e-8)
    soft_sample = sample_gumbel_softmax(logits, temperature, dim, device)
    if hard:
        hard_sample = create_one_hot(soft_sample, dim=dim)
        return (hard_sample - soft_sample).detach() + soft_sample
    else:
        return soft_sample
        
def entropy(in_tensor, marginalize=False):
    if not marginalize:
        b = functional.softmax(in_tensor, dim=1) * functional.log_softmax(in_tensor, dim=1)
        h = -1.0 * b.sum()
    else:
        b = functional.softmax(in_tensor, dim=1).mean(0)
        h = -torch.sum(b*torch.log(b+1e-6))
    return h
    
def marginal_cross_entropy(x, y):
    x = functional.softmax(x, dim=1).mean(0)
    y = functional.softmax(y, dim=1).mean(0)
    h = -torch.sum(x * torch.log(y + 1e-6))
    return h


def create_one_hot(soft_prob, dim):
    indices = torch.argmax(soft_prob, dim=dim)
    hard = functional.one_hot(indices, soft_prob.size()[dim])
    new_axes = tuple(range(dim)) + (len(hard.shape)-1,) + tuple(range(dim, len(hard.shape)-1))
    return hard.permute(new_axes).float()

class KLDivergenceLoss(nn.Module):
    def __init__(self, reduction='none'):
        super(KLDivergenceLoss, self).__init__()
        self.reduction = reduction
    def forward(self, mu, logvar):
        kld = -0.5 * logvar + 0.5 * (torch.exp(logvar) + torch.pow(mu,2)) - 0.5
        if self.reduction == 'mean':
            kld = kld.mean()
        return kld

class CosineDissimilarityLoss(nn.Module):
    def __init__(self, margin=0.0, reduction='mean', eps=1e-6, scale=1.0):
        super(CosineDissimilarityLoss, self).__init__()
        self.margin = margin
        self.eps = eps
        self.reduction = reduction
        self.scale = scale

    def forward(self, rec, trg):
        # noinspection PyTypeChecker
        loss: torch.Tensor = torch.max(functional.cosine_similarity(rec.view(rec.size()[0], rec.size()[1], -1),
                                                                    trg.view(trg.size()[0], trg.size()[1], -1),
                                                                    dim=-1, eps=self.eps) - self.margin,
                                       torch.as_tensor(0.0, device=rec.device)) * self.scale
        return loss if self.reduction == 'none' else (torch.mean(loss) if self.reduction == 'mean'
                                                      else torch.sum(loss))

# http://stackoverflow.com/a/22718321
def mkdir_p(path):
    import os
    import errno
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
