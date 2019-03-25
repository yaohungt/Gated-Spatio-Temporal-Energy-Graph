"""
Adaptive Asynchronous Temporal Fields Base model
"""
import torch.nn as nn
import torch
from torch.autograd import Variable

class BasicModule(nn.Module):
    def __init__(self, inDim, outDim, hidden_dim = 1000, dp_rate = 0.3):
        super(BasicModule, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(inDim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p = dp_rate),
            nn.Linear(hidden_dim, outDim)
            )
        
    def forward(self, x):
        return self.layers(x)
        

class AsyncTFBase(nn.Module):
    def __init__(self, dim, s_classes, o_classes, v_classes, _BaseModule = BasicModule):
        super(AsyncTFBase, self).__init__()
        self.s_classes = s_classes
        self.o_classes = o_classes
        self.v_classes = v_classes
        
        self.num_low_rank = 5

        self.s = nn.Sequential(
            nn.Linear(dim, 1000),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(1000, self.s_classes)
            )
        self.o = nn.Linear(dim, self.o_classes)
        self.v = nn.Linear(dim, self.v_classes)
        
        ### label compatibility matrix
        ## spatio
        self.so_a = _BaseModule(dim, self.s_classes * self.num_low_rank)
        self.so_b = _BaseModule(dim, self.num_low_rank * self.o_classes)
        
        self.ov_a = _BaseModule(dim, self.o_classes * self.num_low_rank) 
        self.ov_b = _BaseModule(dim, self.num_low_rank * self.v_classes) 
        
        self.vs_a = _BaseModule(dim, self.v_classes * self.num_low_rank) 
        self.vs_b = _BaseModule(dim, self.num_low_rank * self.s_classes) 
        
        ## temporal
        self.ss_a = _BaseModule(dim, self.s_classes * self.num_low_rank) 
        self.ss_b = _BaseModule(dim, self.num_low_rank * self.s_classes) 
        
        self.oo_a = _BaseModule(dim, self.o_classes * self.num_low_rank)
        self.oo_b = _BaseModule(dim, self.num_low_rank * self.o_classes)
        
        self.vv_a = _BaseModule(dim, self.v_classes * self.num_low_rank)
        self.vv_b = _BaseModule(dim, self.num_low_rank * self.v_classes) 
        
        self.so_t_a = _BaseModule(dim, self.s_classes * self.num_low_rank) 
        self.so_t_b = _BaseModule(dim, self.num_low_rank * self.o_classes) 
        
        self.ov_t_a = _BaseModule(dim, self.o_classes * self.num_low_rank)
        self.ov_t_b = _BaseModule(dim, self.num_low_rank * self.v_classes)
        
        self.vs_t_a = _BaseModule(dim, self.v_classes * self.num_low_rank)
        self.vs_t_b = _BaseModule(dim, self.num_low_rank * self.s_classes) 
        
        self.os_t_a = _BaseModule(dim, self.o_classes * self.num_low_rank) 
        self.os_t_b = _BaseModule(dim, self.num_low_rank * self.s_classes) 
        
        self.vo_t_a = _BaseModule(dim, self.v_classes * self.num_low_rank)
        self.vo_t_b = _BaseModule(dim, self.num_low_rank * self.o_classes)
        
        self.sv_t_a = _BaseModule(dim, self.s_classes * self.num_low_rank)
        self.sv_t_b = _BaseModule(dim, self.num_low_rank * self.v_classes) 
        
    def forward(self, rgb_feat):
        s = self.s(rgb_feat)
        o = self.o(rgb_feat)
        v = self.v(rgb_feat)
        
        feat = rgb_feat
            
        so_a = self.so_a(feat).view(-1, self.s_classes, self.num_low_rank)
        so_b = self.so_b(feat).view(-1, self.num_low_rank, self.o_classes)
        so = torch.bmm(so_a, so_b)
        
        ov_a = self.ov_a(feat).view(-1, self.o_classes, self.num_low_rank)
        ov_b = self.ov_b(feat).view(-1, self.num_low_rank, self.v_classes)
        ov = torch.bmm(ov_a, ov_b)
        
        vs_a = self.vs_a(feat).view(-1, self.v_classes, self.num_low_rank)
        vs_b = self.vs_b(feat).view(-1, self.num_low_rank, self.s_classes)
        vs = torch.bmm(vs_a, vs_b)
        
        ss_a = self.ss_a(feat).view(-1, self.s_classes, self.num_low_rank)
        ss_b = self.ss_b(feat).view(-1, self.num_low_rank, self.s_classes)
        ss = torch.bmm(ss_a, ss_b)
        
        oo_a = self.oo_a(feat).view(-1, self.o_classes, self.num_low_rank)
        oo_b = self.oo_b(feat).view(-1, self.num_low_rank, self.o_classes)
        oo = torch.bmm(oo_a, oo_b)
        
        vv_a = self.vv_a(feat).view(-1, self.v_classes, self.num_low_rank)
        vv_b = self.vv_b(feat).view(-1, self.num_low_rank, self.v_classes)
        vv = torch.bmm(vv_a, vv_b)
        
        so_t_a = self.so_t_a(feat).view(-1, self.s_classes, self.num_low_rank)
        so_t_b = self.so_t_b(feat).view(-1, self.num_low_rank, self.o_classes)
        so_t = torch.bmm(so_t_a, so_t_b)
        
        ov_t_a = self.ov_t_a(feat).view(-1, self.o_classes, self.num_low_rank)
        ov_t_b = self.ov_t_b(feat).view(-1, self.num_low_rank, self.v_classes)
        ov_t = torch.bmm(ov_t_a, ov_t_b)
        
        vs_t_a = self.vs_t_a(feat).view(-1, self.v_classes, self.num_low_rank)
        vs_t_b = self.vs_t_b(feat).view(-1, self.num_low_rank, self.s_classes)
        vs_t = torch.bmm(vs_t_a, vs_t_b)
        
        os_t_a = self.os_t_a(feat).view(-1, self.o_classes, self.num_low_rank)
        os_t_b = self.os_t_b(feat).view(-1, self.num_low_rank, self.s_classes)
        os_t = torch.bmm(os_t_a, os_t_b)
        
        vo_t_a = self.vo_t_a(feat).view(-1, self.v_classes, self.num_low_rank)
        vo_t_b = self.vo_t_b(feat).view(-1, self.num_low_rank, self.o_classes)
        vo_t = torch.bmm(vo_t_a, vo_t_b)
        
        sv_t_a = self.sv_t_a(feat).view(-1, self.s_classes, self.num_low_rank)
        sv_t_b = self.sv_t_b(feat).view(-1, self.num_low_rank, self.v_classes)
        sv_t = torch.bmm(sv_t_a, sv_t_b)
        
        return s, o, v, so, ov, vs, ss, oo, vv, so_t, ov_t, vs_t, os_t, vo_t, sv_t