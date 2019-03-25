""" Defines the Trainer class which handles train/validation/validation_video
"""
import time
import torch
import itertools
import numpy as np
#from utils import map
from utils import get_predictions, eval_visual_relation
import gc

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(startlr, decay_rate, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = startlr * (0.1 ** (epoch // decay_rate))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy_s(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res[0], res[1], correct[:1].view(-1).float()



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = torch.zeros(*pred.shape)
    for i in range(correct.shape[0]):
        for j in range(correct.shape[1]):
            correct[i, j] = target[j, pred[i, j]] > 0.5

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res[0], res[1], correct[:1].view(-1).float()






def submission_file(ids, outputs, filename):
    """ write list of ids and outputs to filename"""
    with open(filename, 'w') as f:
        for vid, output in zip(ids, outputs):
            scores = ['{:g}'.format(x)
                      for x in output]
            f.write('{} {}\n'.format(vid, ' '.join(scores)))

def gtmat(sizes, target):
    # convert target to a matrix of zeros and ones
    out = torch.zeros(*sizes)
    for i, t in enumerate(target):
        t = t.data[0] if type(t) is torch.Tensor else t
        if len(sizes) == 3:
            out[i, t, :] = 1
        else:
            out[i, t] = 1
    return out.cuda()
    
class Trainer():
    def train(self, loader, base_model, logits_model, criterion, base_optimizer, logits_optimizer, epoch, args):
        adjust_learning_rate(args.lr, args.lr_decay_rate, base_optimizer, epoch)
        adjust_learning_rate(args.lr, args.lr_decay_rate, logits_optimizer, epoch)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        s_top1 = AverageMeter()
        s_top5 = AverageMeter()
        o_top1 = AverageMeter()
        o_top5 = AverageMeter()
        v_top1 = AverageMeter()
        v_top5 = AverageMeter()    
        sov_top1 = AverageMeter()

        # switch to train mode
        base_model.train()
        logits_model.train()
        criterion.train()
        base_optimizer.zero_grad()
        logits_optimizer.zero_grad()

        def part(x): return itertools.islice(x, int(len(x)*args.train_size))
        end = time.time()
        for i, (input, s_target, o_target, v_target, meta) in enumerate(part(loader)):
            gc.collect()
            data_time.update(time.time() - end)
            meta['epoch'] = epoch

            s_target = s_target.long().cuda(async=True)
            o_target = o_target.long().cuda(async=True)
            v_target = v_target.long().cuda(async=True)
            input_var = torch.autograd.Variable(input.cuda())
            s_target_var = torch.autograd.Variable(s_target)
            o_target_var = torch.autograd.Variable(o_target)
            v_target_var = torch.autograd.Variable(v_target)
            
            feat = base_model(input_var)
            s, o, v, so, ov, vs, ss, oo, vv, so_t, ov_t, vs_t, os_t, vo_t, sv_t = logits_model(feat)
            
            s_output, o_output, v_output, loss = criterion(*((s, o, v, so, ov, vs, ss, oo, vv, so_t, ov_t, vs_t, os_t, vo_t, sv_t) + (s_target_var, o_target_var, v_target_var, meta)))
            
            s_prec1, s_prec5, s_prec1_output = accuracy_s(s_output.data, s_target, topk=(1, 5))
            o_prec1, o_prec5, o_prec1_output = accuracy(o_output.data, o_target, topk=(1, 5))
            v_prec1, v_prec5, v_prec1_output = accuracy(v_output.data, v_target, topk=(1, 5))

            sov_prec1 = s_prec1_output.cpu() * o_prec1_output * v_prec1_output
            sov_prec1 = sov_prec1.sum(0, keepdim=True)
            sov_prec1 = sov_prec1.mul_(100.0 / input.size(0))
            
            s_top1.update(s_prec1[0], input.size(0))
            s_top5.update(s_prec5[0], input.size(0))
            o_top1.update(o_prec1[0], input.size(0))
            o_top5.update(o_prec5[0], input.size(0))
            v_top1.update(v_prec1[0], input.size(0))
            v_top5.update(v_prec5[0], input.size(0))
            sov_top1.update(sov_prec1[0], input.size(0))
            
            losses.update(loss.data[0], input.size(0))
            loss.backward()
            if i % args.accum_grad == args.accum_grad-1:
                #print('updating parameters')
                if False:
                    base_optimizer.step()
                    base_optimizer.zero_grad()
                logits_optimizer.step()
                logits_optimizer.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}({3})]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'S_Prec@1 {s_top1.val:.3f} ({s_top1.avg:.3f})\t'
                      'S_Prec@5 {s_top5.val:.3f} ({s_top5.avg:.3f})\t'
                      'O_Prec@1 {o_top1.val:.3f} ({o_top1.avg:.3f})\t'
                      'O_Prec@5 {o_top5.val:.3f} ({o_top5.avg:.3f})\t'
                      'V_Prec@1 {v_top1.val:.3f} ({v_top1.avg:.3f})\t'
                      'V_Prec@5 {v_top5.val:.3f} ({v_top5.avg:.3f})\t'
                      'SOV_Prec@1 {sov_top1.val:.3f} ({sov_top1.avg:.3f})'.format(
                          epoch, i, int(
                              len(loader)*args.train_size), len(loader),
                          batch_time=batch_time, data_time=data_time, loss=losses, s_top1=s_top1, s_top5=s_top5, o_top1=o_top1, o_top5=o_top5, v_top1=v_top1, v_top5=v_top5, sov_top1 = sov_top1))
        return s_top1.avg, s_top5.avg, o_top1.avg, o_top5.avg, v_top1.avg, v_top5.avg, sov_top1.avg

    def validate(self, loader, base_model, logits_model, criterion, epoch, args):
        with torch.no_grad():
            batch_time = AverageMeter()
            losses = AverageMeter()
            s_top1 = AverageMeter()
            s_top5 = AverageMeter()
            o_top1 = AverageMeter()
            o_top5 = AverageMeter()
            v_top1 = AverageMeter()
            v_top5 = AverageMeter()  
            sov_top1 = AverageMeter()

            # switch to evaluate mode
            base_model.eval()
            logits_model.eval()
            criterion.eval()

            def part(x): return itertools.islice(x, int(len(x)*args.val_size))
            end = time.time()
            for i, (input, s_target, o_target, v_target, meta) in enumerate(part(loader)):
                gc.collect()
                meta['epoch'] = epoch
                s_target = s_target.long().cuda(async=True)
                o_target = o_target.long().cuda(async=True)
                v_target = v_target.long().cuda(async=True)
                input_var = torch.autograd.Variable(input.cuda())
                s_target_var = torch.autograd.Variable(s_target)
                o_target_var = torch.autograd.Variable(o_target)
                v_target_var = torch.autograd.Variable(v_target)
                
                feat = base_model(input_var)
                s, o, v, so, ov, vs, ss, oo, vv, so_t, ov_t, vs_t, os_t, vo_t, sv_t = logits_model(feat)
                
                s_output, o_output, v_output, loss = criterion(*((s, o, v, so, ov, vs, ss, oo, vv, so_t, ov_t, vs_t, os_t, vo_t, sv_t) + (s_target_var, o_target_var, v_target_var, meta)))
                
                s_prec1, s_prec5, s_prec1_output = accuracy_s(s_output.data, s_target, topk=(1, 5))
                o_prec1, o_prec5, o_prec1_output = accuracy(o_output.data, o_target, topk=(1, 5))
                v_prec1, v_prec5, v_prec1_output = accuracy(v_output.data, v_target, topk=(1, 5))      
             
                sov_prec1 = s_prec1_output.cpu() * o_prec1_output * v_prec1_output
                sov_prec1 = sov_prec1.sum(0, keepdim=True)
                sov_prec1 = sov_prec1.mul_(100.0 / input.size(0))
                
                s_top1.update(s_prec1[0], input.size(0))
                s_top5.update(s_prec5[0], input.size(0))
                o_top1.update(o_prec1[0], input.size(0))
                o_top5.update(o_prec5[0], input.size(0))
                v_top1.update(v_prec1[0], input.size(0))
                v_top5.update(v_prec5[0], input.size(0))
                sov_top1.update(sov_prec1[0], input.size(0))
                
            
                losses.update(loss.data[0], input.size(0))

                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    print('Test: [{0}/{1} ({2})]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'S_Prec@1 {s_top1.val:.3f} ({s_top1.avg:.3f})\t'
                          'S_Prec@5 {s_top5.val:.3f} ({s_top5.avg:.3f})\t'
                          'O_Prec@1 {o_top1.val:.3f} ({o_top1.avg:.3f})\t'
                          'O_Prec@5 {o_top5.val:.3f} ({o_top5.avg:.3f})\t'
                          'V_Prec@1 {v_top1.val:.3f} ({v_top1.avg:.3f})\t'
                          'V_Prec@5 {v_top5.val:.3f} ({v_top5.avg:.3f})\t'
                          'SOV_Prec@1 {sov_top1.val:.3f} ({sov_top1.avg:.3f})'.format(
                              i, int(len(loader)*args.val_size), len(loader),
                              batch_time=batch_time, loss=losses, s_top1=s_top1, s_top5=s_top5, o_top1=o_top1, o_top5=o_top5, v_top1=v_top1, v_top5=v_top5, sov_top1 = sov_top1))
            return s_top1.avg, s_top5.avg, o_top1.avg, o_top5.avg, v_top1.avg, v_top5.avg, sov_top1.avg

    def validate_video(self, loader, base_model, logits_model, criterion, epoch, args):
        """ Run video-level validation on the Charades test set"""
        with torch.no_grad():
            batch_time = AverageMeter()
            ids = []

            sov_prediction = dict()
            
            # switch to evaluate mode
            base_model.eval()
            logits_model.eval()
            criterion.eval()

            end = time.time()
            for i, (input, s_target, o_target, v_target, meta) in enumerate(loader):
                gc.collect()
                meta['epoch'] = epoch
                s_target = s_target.long().cuda(async=True)
                o_target = o_target.long().cuda(async=True)
                v_target = v_target.long().cuda(async=True)
                input_var = torch.autograd.Variable(input.cuda())
                s_target_var = torch.autograd.Variable(s_target)
                o_target_var = torch.autograd.Variable(o_target)
                v_target_var = torch.autograd.Variable(v_target)
                
                feat = base_model(input_var)
                s, o, v, so, ov, vs, ss, oo, vv, so_t, ov_t, vs_t, os_t, vo_t, sv_t = logits_model(feat)
                
                s_output, o_output, v_output, loss = criterion(*((s, o, v, so, ov, vs, ss, oo, vv, so_t, ov_t, vs_t, os_t, vo_t, sv_t) + (s_target_var, o_target_var, v_target_var, meta)), synchronous=True)
                
                # store predictions
                s_output_video = s_output.max(dim=0)[0]
                o_output_video = o_output.max(dim=0)[0]
                v_output_video = v_output.max(dim=0)[0]
                
                sov_prediction[meta['id'][0]] = get_predictions(s_output_video.data.cpu().numpy(),o_output_video.data.cpu().numpy(), v_output_video.data.cpu().numpy() )
                
                ids.append(meta['id'][0])
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    print('Test2: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                              i, len(loader), batch_time=batch_time))
                    
            sov_mAP, sov_rec_at_n, sov_mprec_at_n = eval_visual_relation(prediction=sov_prediction, groundtruth_path=args.groundtruth-lookup)
            print(' * sov_mAP {:.3f}'.format(sov_mAP))
            print(' * sov_rec_at_n', sov_rec_at_n)
            print(' * sov_mprec_at_n', sov_mprec_at_n)
            return sov_mAP, sov_rec_at_n, sov_mprec_at_n 
