# pylint: disable=W0221,E1101
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from random import random
from models.layers.VerboseGradients import VerboseGradients
from models.layers.BalanceLabels import BalanceLabels

import numpy as np


def unit(x):
    # normalize tensor in log space to have unit sum for each row
    minx, _ = x.max(1)
    z = (x - minx[:, None]).exp().sum(1).log() + minx
    return x - z[:, None]


def lse(x, dim=None, keepdim=False):
    # log sum exp @alshedivat
    return (x - F.log_softmax(x)).sum(dim, keepdim=keepdim)


def sme(x, y, dim=None, keepdim=False):
    # Sum mul exp
    return (x * torch.exp(y)).sum(dim, keepdim=keepdim)


def axb(a, x, b):
    # a and b are batched vectors, X is batched matrix
    # returns a^t * X * b
    xb = torch.bmm(x, b[:, :, None])
    return (a * xb.squeeze()).sum(1)


def avg(iterator, weight=1.):
    # compounding weight
    item, w = next(iterator)
    total = item.clone() * w
    n = 1.
    for i, (item, w) in enumerate(iterator):
        w1 = 1. * weight**(i + 1)
        total += item * w1 * w
        n += w1
    return total / n


def winsmooth(mat, kernelsize=1):
    #print('applying smoothing with kernelsize {}'.format(kernelsize))
    mat.detach()
    n = mat.shape[0]
    out = mat.clone()
    for m in range(n):
        a = max(0, m - kernelsize)
        b = min(n - 1, m + kernelsize)
        out[m, :] = mat[a:b + 1, :].mean(0)
    return out


def gtmat(sizes, target):
    # convert target to a matrix of zeros and ones
    out = torch.zeros(*sizes)
    for i, t in enumerate(target):
        t = t.data[0] if type(t) is torch.Tensor else t
        if len(sizes) == 3:
            out[i, t, :] = 1
        else:
            out[i, t] = 1
    if type(target) is Variable:
        return Variable(out.cuda())
    else:
        return out.cuda()


def nll_loss(soft_target, logdist, reduce=True):
    # @Hongyi_Zhang
    # assumes soft_target is normalized to 1 and between [0,1]
    # logdist is a (normalized) log distribution
    logdist = unit((logdist.exp() + 0.00001).log())  # for numerical stability
    if soft_target.dim() == 3:
        out = (-soft_target * logdist).sum(2).sum(1)
    else:
        out = (-soft_target * logdist).sum(1)
    if reduce:
        return out.mean()
    else:
        return out


class MessagePassing(object):
    # Class for keeping track of messages across frames
    def __init__(self, maxsize, w_temporal, w_spatio, decay, sigma, ns, no, nv):
        super(MessagePassing, self).__init__()
        self.maxsize = maxsize
        self.w_temporal = w_temporal
        self.w_spatio = w_spatio
        self.decay = decay
        self.sigma = sigma
        self.s_storage = {}
        self.s_storage_gt = {}
        self.o_storage = {}
        self.o_storage_gt = {}
        self.v_storage = {}
        self.v_storage_gt = {}
        self.training = self.training if hasattr(self, 'training') else True
        self.ns = ns
        self.no = no
        self.nv = nv

    def mget(self, idtime, s_size, o_size, v_size, s_storage, o_storage, v_storage, cond=lambda t, t0: True, kernel=lambda t, t0: 1):
        # get message using condition on the timestamps
        def meta(ids, t0, size, storage):
            try:
                return avg(((y, kernel(t, t0)) for t, y in storage[ids]
                            if cond(t, t0)), 1. / self.decay)
            except (StopIteration, KeyError):
                return torch.zeros(size)
        s_out = [meta(ids, time, s_size, s_storage) for ids, time in idtime]
        o_out = [meta(ids, time, o_size, o_storage) for ids, time in idtime]
        v_out = [meta(ids, time, v_size, v_storage) for ids, time in idtime]
        return Variable(torch.stack(s_out, 0).cuda()), Variable(torch.stack(o_out, 0).cuda()), Variable(torch.stack(v_out, 0).cuda())

    def get_msg(self, idtime, time='past', s_storage=None, o_storage=None, v_storage=None):
        s_storage = self.s_storage if s_storage is None else s_storage
        o_storage = self.o_storage if o_storage is None else o_storage
        v_storage = self.v_storage if v_storage is None else v_storage
        cond = lambda t, t0: t < t0 if time == 'past' else t > t0
        kernel = lambda t, t0: math.exp(-float(t - t0)**2 / (2 * self.sigma**2))
        return self.mget(idtime, self.ns, self.no, self.nv, s_storage, o_storage, v_storage, cond, kernel) 

    def get_gt_msg(self, idtime, time='past'):
        return self.get_msg(idtime, time, self.s_storage_gt, self.o_storage_gt, self.v_storage_gt)

    def mset(self, s_msg, o_msg, v_msg, idtime, s_storage, o_storage, v_storage):
        # keep a queue of size maxsize for each id
        # messages are stored in normal space
        # queue for each id is stored in the order in which the messages were stored
        for s_m, o_m, v_m, (ids, time) in sorted(zip(s_msg, o_msg, v_msg, idtime), key=lambda x: random()):
            if ids not in s_storage:
                s_storage[ids] = []
            if ids not in o_storage:
                o_storage[ids] = []
            if ids not in v_storage:
                v_storage[ids] = []
            
            s_data = s_m if type(s_m) is not torch.Tensor else s_m.data.cpu()
            o_data = o_m if type(o_m) is not torch.Tensor else o_m.data.cpu()
            v_data = v_m if type(v_m) is not torch.Tensor else v_m.data.cpu()
            
            s_storage[ids].append((time, s_data))
            o_storage[ids].append((time, o_data))
            v_storage[ids].append((time, v_data))
            
            if len(s_storage[ids]) > self.maxsize:
                del s_storage[ids][0]
            if len(o_storage[ids]) > self.maxsize:
                del o_storage[ids][0]
            if len(v_storage[ids]) > self.maxsize:
                del v_storage[ids][0]

    def set_msg(self, qs, qo, qv, idtime):
        self.mset(qs, qo, qv, idtime, self.s_storage, self.o_storage, self.v_storage)

    def set_gt_msg(self, s_target, o_target, v_target, idtime):
        s_x = s_target.data.cpu()
        o_x = o_target.data.cpu()
        v_x = v_target.data.cpu()
        self.mset(s_x, o_x, v_x, idtime, self.s_storage_gt, self.o_storage_gt, self.v_storage_gt)


class AsyncTFCriterion(nn.Module, MessagePassing):
    def __init__(self, args):
        memory_size = 20
        w_temporal = 0.1
        w_spatio = 0.1
        memory_decay = 1.0
        sigma = 300
        MessagePassing.__init__(self, memory_size, w_temporal, w_spatio, memory_decay, sigma, args.s_class, args.o_class, args.v_class)
        nn.Module.__init__(self)
        self.msg_n = 5
        
        self.cross_loss = nn.CrossEntropyLoss() # for s
        self.bce_loss = nn.BCEWithLogitsLoss() # for c, o, v 
        
        self.BalanceLabels = BalanceLabels()
        self.winsmooth = 1

    def forward(self, s, o, v, so, ov, vs, ss, oo, vv, so_t, ov_t, vs_t, os_t, vo_t, sv_t, s_target, o_target, v_target, id_time, n=1, synchronous=False):
        if o_target.dim() == 1:
            print('converting Nx1 target to NxC')
            o_target = Variable(gtmat(o.shape, o_target.data.long()))
        if v_target.dim() == 1:
            print('converting Nx1 target to NxC')
            v_target = Variable(gtmat(v.shape, v_target.data.long()))
        o_target = o_target.float()
        v_target = v_target.float()
        idtime = list(zip(id_time['id'], id_time['time']))

        
        s_msg, o_msg, v_msg  = self.get_msg(idtime, 'past')
        s_fmsg, o_fmsg, v_fmsg  = self.get_msg(idtime, 'future')

        s_loss = self.cross_loss(s, s_target)
        _qs = torch.nn.Softmax(dim = 1)(s)
        o_loss = self.bce_loss(o, o_target) 
        _qo = torch.nn.Sigmoid()(o)
        v_loss = self.bce_loss(v, v_target)
        _qv = torch.nn.Sigmoid()(v)
        
        qs_before_softmax = s.clone()
        

        qs_before_softmax += torch.bmm(s_msg.unsqueeze(1), ss).squeeze() * self.w_temporal
        qs_before_softmax += torch.bmm(ss, s_fmsg.unsqueeze(2)).squeeze() * self.w_temporal
        qs_before_softmax += torch.bmm(o_msg.unsqueeze(1), os_t).squeeze() * self.w_temporal 
        qs_before_softmax += torch.bmm(so_t, o_fmsg.unsqueeze(2)).squeeze() * self.w_temporal
        qs_before_softmax += torch.bmm(v_msg.unsqueeze(1), vs_t).squeeze() * self.w_temporal
        qs_before_softmax += torch.bmm(sv_t, v_fmsg.unsqueeze(2)).squeeze() * self.w_temporal
        qs_before_softmax += torch.bmm(so, _qo.unsqueeze(2)).squeeze() * self.w_spatio 
        qs_before_softmax += torch.bmm(_qv.unsqueeze(1), vs).squeeze() * self.w_spatio
        
        s_loss += self.cross_loss(qs_before_softmax, s_target)
        qs = torch.nn.Softmax(dim = 1)(qs_before_softmax)
        
        qo_before_sigmoid = o.clone()

        qo_before_sigmoid += torch.bmm(o_msg.unsqueeze(1), oo).squeeze() * self.w_temporal 
        qo_before_sigmoid += torch.bmm(oo, o_fmsg.unsqueeze(2)).squeeze() * self.w_temporal
        qo_before_sigmoid += torch.bmm(v_msg.unsqueeze(1), vo_t).squeeze() * self.w_temporal
        qo_before_sigmoid += torch.bmm(ov_t, v_fmsg.unsqueeze(2)).squeeze() * self.w_temporal
        qo_before_sigmoid += torch.bmm(s_msg.unsqueeze(1), so_t).squeeze() * self.w_temporal 
        qo_before_sigmoid += torch.bmm(os_t, s_fmsg.unsqueeze(2)).squeeze() * self.w_temporal 
        qo_before_sigmoid += torch.bmm(_qs.unsqueeze(1), so).squeeze() * self.w_spatio
        qo_before_sigmoid += torch.bmm(ov, _qv.unsqueeze(2)).squeeze() * self.w_spatio
        
        o_loss += self.bce_loss(qo_before_sigmoid, o_target) 
        qo = torch.nn.Sigmoid()(qo_before_sigmoid)        
        
        qv_before_sigmoid = v.clone()
            
        qv_before_sigmoid += torch.bmm(v_msg.unsqueeze(1), vv).squeeze() * self.w_temporal 
        qv_before_sigmoid += torch.bmm(vv, v_fmsg.unsqueeze(2)).squeeze() * self.w_temporal 
        qv_before_sigmoid += torch.bmm(s_msg.unsqueeze(1), sv_t).squeeze() * self.w_temporal 
        qv_before_sigmoid += torch.bmm(vs_t, s_fmsg.unsqueeze(2)).squeeze() * self.w_temporal 
        qv_before_sigmoid += torch.bmm(o_msg.unsqueeze(1), ov_t).squeeze() * self.w_temporal
        qv_before_sigmoid += torch.bmm(vo_t, o_fmsg.unsqueeze(2)).squeeze() * self.w_temporal 
        qv_before_sigmoid += torch.bmm(vs, _qs.unsqueeze(2)).squeeze() * self.w_spatio
        qv_before_sigmoid += torch.bmm(_qo.unsqueeze(1), ov).squeeze() * self.w_spatio 
        
        v_loss += self.bce_loss(qv_before_sigmoid, v_target) 
        qv = torch.nn.Sigmoid()(qv_before_sigmoid)        

        
        # self.set_msg(qs, qo, qv, idtime)
        self.set_msg(_qs, _qo, _qv, idtime)
        
        loss = s_loss + o_loss + v_loss
        if not synchronous or n > self.msg_n:
            s_out, o_out, v_out = qs.clone(), qo.clone(), qv.clone()
            if synchronous:
                s_out = winsmooth(s_out, kernelsize=self.winsmooth)
                o_out = winsmooth(o_out, kernelsize=self.winsmooth)
                v_out = winsmooth(v_out, kernelsize=self.winsmooth)
            return s_out, o_out, v_out, loss
        else:
            return self.forward(s, o, v, so, ov, vs, ss, oo, vv, so_t, ov_t, vs_t, os_t, vo_t, sv_t, s_target, o_target, v_target, id_time, n=n + 1, synchronous=synchronous)
