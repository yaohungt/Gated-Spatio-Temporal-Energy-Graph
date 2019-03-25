import numpy as np
import pickle
from collections import defaultdict



def eval_tagging_scores(gt_relations, pred_relations):
    # ignore trajectories
    gt_triplets = set(tuple(r) for r in gt_relations)
    pred_triplets = []
    hit_scores = []
    for s, triplet in pred_relations:
        if not triplet in pred_triplets:
            pred_triplets.append(triplet)
            hit_scores.append(s)
    hit_scores = np.asarray(hit_scores)
    for i, t in enumerate(pred_triplets):
        if not t in gt_triplets:
            hit_scores[i] = -np.inf
    tp = np.isfinite(hit_scores)
    fp = ~tp
    cum_tp = np.cumsum(tp).astype(np.float32)
    cum_fp = np.cumsum(fp).astype(np.float32)
    rec = cum_tp / np.maximum(len(gt_triplets), np.finfo(np.float32).eps)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float32).eps)
    return prec, rec, hit_scores



def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC ap given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct ap calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def eval_visual_relation(prediction, groundtruth_path = '', rec_nreturns=[50, 100], prec_nreturns=[1, 5, 10]):
    """ evaluate visual relation tagging.
    """
    with open(groundtruth_path, 'rb') as file:
        groundtruth = pickle.load(file)
    print('evaluating...')
    video_ap = dict()
    tot_scores = defaultdict(list)
    tot_tp = defaultdict(list)
    prec_at_n = defaultdict(list)
    tot_gt_relations = 0
    for vid, gt_relations in groundtruth.items():
        predict_relations = prediction[vid]
        tag_prec, tag_rec, tag_scores = eval_tagging_scores(gt_relations, predict_relations)
        # record per video evaluation results
        video_ap[vid] = voc_ap(tag_rec, tag_prec)
        tp = np.isfinite(tag_scores)
        for nre in rec_nreturns:
            cut_off = min(nre, tag_scores.size)
            tot_scores[nre].append(tag_scores[:cut_off])
            tot_tp[nre].append(tp[:cut_off])
        for nre in prec_nreturns:
            cut_off = min(nre, tag_scores.size)
            prec_at_n[nre].append(tag_prec[cut_off - 1])
        tot_gt_relations += len(gt_relations)
    # calculate mean ap for detection
    mAP = np.mean(list(video_ap.values()))
    # calculate recall for detection
    rec_at_n = dict()
    for nre in rec_nreturns:
        scores = np.concatenate(tot_scores[nre])
        tps = np.concatenate(tot_tp[nre])
        sort_indices = np.argsort(scores)[::-1]
        tps = tps[sort_indices]
        cum_tp = np.cumsum(tps).astype(np.float32)
        rec = cum_tp / np.maximum(tot_gt_relations, np.finfo(np.float32).eps)
        rec_at_n[nre] = rec[-1]
    # calculate mean precision for tagging
    mprec_at_n = dict()
    for nre in prec_nreturns:
        mprec_at_n[nre] = np.mean(prec_at_n[nre])
    return mAP, rec_at_n, mprec_at_n

def get_predictions(inference_s, inference_o, inference_v):
    top_s_ind = np.argsort(inference_s)[-10:] 
    top_o_ind = np.argsort(inference_o)[-10:] 
    top_v_ind = np.argsort(inference_v)[-10:] 

    score = inference_s[top_s_ind, None, None] + inference_o[None, top_o_ind, None] + inference_v[None, None, top_v_ind]   
    top_flat_ind = np.argsort(score, axis = None)[-200:]
    top_score = score.ravel()[top_flat_ind]
    
    top_s, top_o, top_v = np.unravel_index(top_flat_ind, score.shape)
    
    predictions = [(top_score[j], 
        (top_s_ind[top_s[j]], top_o_ind[top_o[j]], top_v_ind[top_v[j]])) 
        for j in range(top_score.size)]
    
    predictions = sorted(predictions, key=lambda x: x[0], reverse=True) 
    
    return predictions
