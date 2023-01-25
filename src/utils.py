import numpy as np
import logging
import torch
logger = logging.getLogger(__name__)
import sys
import os
import json
import torch.distributed as dist
import torch.nn.functional as F

mask = None

def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)

def init_logger(is_main=True, is_distributed=False, filename=None):
    if is_distributed:
        torch.distributed.barrier()
    handlers = [logging.StreamHandler(sys.stdout)]
    if filename is not None:
        handlers.append(logging.FileHandler(filename=filename))
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main else logging.WARN,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )

    return logger

def load(model, dir_path, opt, reset_params=False):
    epoch_path = os.path.realpath(dir_path)
    optimizer_path = os.path.join(epoch_path, "optimizer.pth.tar")
    model_path = os.path.join(epoch_path, "model.pth.tar")
    logger.info("Loading %s" % epoch_path)
    ParetoGNN_config = json.load(open(epoch_path + '/ParetoGNN_config.json'))
    model.load_state_dict(torch.load(model_path))
    opt.ParetoGNN_config = ParetoGNN_config
    logger.info("loading checkpoint %s" %optimizer_path)
    checkpoint = torch.load(optimizer_path, map_location=opt.device)
    opt_checkpoint = checkpoint["opt"]
    step = checkpoint["step"]

    if not reset_params:
        optimizer, scheduler = set_optim(opt_checkpoint, model)
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        optimizer, scheduler = set_optim(opt, model)

    return model, optimizer, scheduler, opt_checkpoint, step


def save(model, optimizer, scheduler, step, opt, dir_path, name):
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save = model_to_save.big_model
    path = os.path.join(dir_path, "checkpoint")
    epoch_path = os.path.join(path, name) #"step-%s" % step)
    os.makedirs(epoch_path, exist_ok=True)
    fp = os.path.join(epoch_path, "model.pth.tar")
    torch.save(model_to_save.state_dict(), fp)
    with open('{}/ParetoGNN_config.json'.format(epoch_path), 'w') as fp: json.dump(opt.ParetoGNN_config, fp, sort_keys=True, indent=4)
    # fp = os.path.join(epoch_path, "optimizer.pth.tar")
    # checkpoint = {
    #     "step": step,
    #     "optimizer": optimizer.state_dict(),
    #     "scheduler": scheduler.state_dict(),
    #     "opt": opt,
    # }
    # torch.save(checkpoint, fp)


class WarmupLinearScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, scheduler_steps, min_ratio, fixed_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.scheduler_steps = scheduler_steps
        self.min_ratio = min_ratio
        self.fixed_lr = fixed_lr
        super(WarmupLinearScheduler, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return (1 - self.min_ratio)*step/float(max(1, self.warmup_steps)) + self.min_ratio

        if self.fixed_lr:
            return 1.0

        return max(0.0,
            1.0 + (self.min_ratio - 1) * (step - self.warmup_steps)/float(max(1.0, self.scheduler_steps - self.warmup_steps)),
        )

class FixedScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, last_epoch=-1):
        super(FixedScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)
    def lr_lambda(self, step):
        return 1.0

def set_optim(opt, model):
    if opt.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    elif opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    if opt.scheduler == 'fixed':
        scheduler = FixedScheduler(optimizer)
    elif opt.scheduler == 'linear':
        if opt.scheduler_steps is None:
            scheduler_steps = opt.total_steps
        else:
            scheduler_steps = opt.scheduler_steps
        scheduler = WarmupLinearScheduler(optimizer, warmup_steps=opt.warmup_steps, scheduler_steps=scheduler_steps, min_ratio=0., fixed_lr=opt.fixed_lr)
    return optimizer, scheduler

def average_main(x, opt):
    if not opt.is_distributed:
        return x
    if opt.world_size > 1:
        dist.reduce(x, 0, op=dist.ReduceOp.SUM)
        if opt.is_main:
            x = x / opt.world_size
    return x


def sum_main(x, opt):
    if not opt.is_distributed:
        return x
    if opt.world_size > 1:
        dist.reduce(x, 0, op=dist.ReduceOp.SUM)
    return x


def load_pretrained(model, dir_path, opt=None):
    epoch_path = os.path.realpath(dir_path)
    model_path = os.path.join(epoch_path, "model.pth.tar")
    logger.info("Loading %s" % epoch_path)
    if opt != None:
        model.load_state_dict(torch.load(model_path, map_location=opt.device))
    else:
        model.load_state_dict(torch.load(model_path))
    return model


def accuracy(logits, labels):
    _, pred = logits.topk(1, dim=1)
    n_correct = torch.sum(pred.t().squeeze() == labels)
    return n_correct/len(labels)

def constrastive_loss(input1, input2, in_batch_neg=True, temperature=0.3):

    n_samples = input1.shape[0]
    # input1 = F.normalize(input1, dim=1)
    # input2 = F.normalize(input2, dim=1)
    
    if in_batch_neg:
        input = torch.cat([input1, input2], dim=0)
        sim = torch.exp(sim_matrix(input, input)/temperature)
        sim = sim - torch.diag(sim.diagonal())
        numerator = torch.cat([torch.diagonal(sim, offset=n_samples),
                               torch.diagonal(sim, offset=-n_samples)])
        denominator = sim.sum(1)
        loss = (numerator/denominator).sum()
    else:
        sim = sim_matrix(input1, input2.t())/temperature
        numerator = torch.exp(torch.diagonal(sim))
        denominator = torch.exp(sim)
        denominator = denominator.sum(1)
        loss = (numerator/denominator).sum()

    return -torch.log((1.0 / n_samples) * loss)

def barlow_twins(input1, input2, lambda_=0.01):

    def off_diagonal(x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    n_samples = input1.shape[0]
    input1_norm = (input1 - input1.mean(0)) / input1.std(0)
    input2_norm = (input2 - input2.mean(0)) / input2.std(0)
    c = (input1_norm.t() @ input2_norm) / n_samples
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    loss = on_diag + lambda_ * off_diag
    return loss

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt