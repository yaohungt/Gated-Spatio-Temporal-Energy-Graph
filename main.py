#!/usr/bin/env python

"""Charades activity recognition baseline code
   Can be run directly or throught config scripts under exp/

   Gunnar Sigurdsson, 2018
""" 
import torch
import numpy as np
import random
import train
from models import create_model
from datasets import get_dataset
import checkpoints
from opts import parse
from utils import tee


def seed(manualseed):
    random.seed(manualseed)
    np.random.seed(manualseed)
    torch.manual_seed(manualseed)
    torch.cuda.manual_seed(manualseed)


best_mAP = 0
def main():
    global opt, best_mAP
    opt = parse()
    tee.Tee(opt.cache+'/log.txt')
    print(vars(opt))
    seed(opt.manual_seed)

    base_model, logits_model, criterion, base_optimizer, logits_optimizer = create_model(opt)
    if opt.resume: best_mAP = checkpoints.load(opt, base_model, logits_model, base_optimizer, logits_optimizer)
    print(logits_model)
    trainer = train.Trainer()
    train_loader, val_loader, valvideo_loader = get_dataset(opt)
    
    if opt.evaluate:
        trainer.validate(val_loader, base_model, logits_model, criterion, -1, opt)
        trainer.validate_video(valvideo_loader, base_model, logits_model, criterion, -1, opt)
        return

    for epoch in range(opt.start_epoch, opt.epochs):
        if opt.distributed:
            trainer.train_sampler.set_epoch(epoch)
        s_top1,s_top5,o_top1,o_top5,v_top1,v_top5, sov_top1 = trainer.train(train_loader, base_model, logits_model, criterion, base_optimizer, logits_optimizer, epoch, opt)
        s_top1val,s_top5val,o_top1val,o_top5val,v_top1val,v_top5val, sov_top1val = trainer.validate(val_loader, base_model, logits_model,  criterion, epoch, opt)
        sov_mAP, sov_rec_at_n, sov_mprec_at_n  = trainer.validate_video(valvideo_loader, base_model, logits_model, criterion, epoch, opt)
        is_best = sov_mAP > best_mAP
        best_mAP = max(sov_mAP, best_mAP)
        scores = {'s_top1':s_top1,'s_top5':s_top5,'o_top1':o_top1,'o_top5':o_top5,'v_top1':v_top1,'v_top5':v_top5,'sov_top1':sov_top1,'s_top1val':s_top1val,'s_top5val':s_top5val,'o_top1val':o_top1val,'o_top5val':o_top5val,'v_top1val':v_top1val,'v_top5val':v_top5val,'sov_top1val':sov_top1val,'mAP':sov_mAP,'sov_rec_at_n':sov_rec_at_n,'sov_mprec_at_n':sov_mprec_at_n}
        checkpoints.save(epoch, opt, base_model, logits_model, base_optimizer, logits_optimizer, is_best, scores)


if __name__ == '__main__':
    main()
