from __future__ import annotations

import torch
from tqdm.auto import tqdm
from torchvision.ops import box_iou
from .infer import Decode_Predictions

@torch.no_grad ()
def Evaluate_Loss (model, criterion, loader, device, max_batches = None):
    was_training = model.training
    model.eval ()
    totals = {}
    batches  =0
    
    for images, targets in tqdm (loader, desc = "val", leave = False):
        images = images.to (device, non_blocking = True)
        moved = {k : v.to (device, non_blocking = True) for k, v in targets.items ()}
        outputs = model (images)
        _, stats = criterion (outputs, moved)
        
        for key, value in stats.items ():
            totals [key] = totals.get (key, 0.0) + float (value)
        
        batches += 1
        
        if max_batches is not None and batches >= max_batches:
            break
        
    if was_training:
        model.train ()
        
    if batches == 0:
        return {"loss" : float ("inf")}
    
    return {key : value / batches for key, value in totals.items ()}

@torch.no_grad ()
def Evaluate_Prediction_Counts (model, loader, device, conf_threshold, max_batches = 10):
    was_training = model.training
    model.eval ()
    images_seen = 0
    predictions_seen = 0
    
    for images, _ in tqdm (loader, desc = "predict-val", leave = False):
        images = images.to (device, non_blocking = True)
        outputs = model (images)
        preds = Decode_Predictions (outputs, conf_threshold = conf_threshold)
        images_seen += len (preds)
        predictions_seen += sum (len (p) for p in preds)
        
        if images_seen >= max_batches * loader.batch_size:
            break
        
    if was_training:
        model.train ()
        
    return {"avg_predictions_per_image" : predictions_seen / max (1, images_seen)}

@torch.no_grad ()
def Evaluate_Detection_Metrics (model, loader, device, num_classses, image_size, ap_conf_threshold = 0.001, metric_conf_threshold = 0.25, nms_iou_threshold = 0.45, match_iou_threshold = 0.5, max_det = 100, max_batches = None):
    was_training = model.training
    model.eval ()
    predictions = []
    ground_truths = []
    batches = 0
    
    for images, targets in tqdm (loader, desc = "detect-val", leave = False):
        images = images.to (device, non_blocking = True)
        outputs = model (images)
        batch_predictions = Decode_Predictions (outputs, conf_threshold = ap_conf_threshold, iou_threshold = nms_iou_threshold, max_detections = max_det, image_size = image_size)
        batch_index = targets ["batch_index"]
        
        for images_index, pred in enumerate (batch_predictions):
            mask = batch_index == images_index
            predictions.append (pred.detach ().cpu ())
            ground_truths.append ({
                "boxes": targets ["boxes"][mask].detach ().cpu (),
                "labels": targets ["labels"][mask].detach ().cpu (),
            })
            
        batches += 1
        
        if max_batches is not None and batches >= max_batches:
            break
        
    if was_training:
        model.train ()
        
    return Detection_Metrics_From_Predictions (predictions, ground_truths, num_classes = num_classses, metric_conf_threshold = metric_conf_threshold, match_iou_threshold = match_iou_threshold)

def Detection_Metrics_From_Predictions (predictions, ground_truths, num_classes, metric_conf_threshold = 0.25, match_iou_threshold = 0.5, iou_thresholds = None):
    if len (predictions) != len (ground_truths):
        raise ValueError ("Predictions and ground truths must have the same length.")
    
    thresholds = iou_thresholds or [round (0.5 + 0.05 * i, 2) for i in range (10)]
    ap_by_threshold = {threshold: _mean_average_precision_at_iou (predictions, ground_truths, num_classes, threshold) for threshold in thresholds}
    counts = _precision_recall_counts (predictions, ground_truths, num_classes, metric_conf_threshold = metric_conf_threshold, match_iou_threshold = match_iou_threshold)
    total_predictions = sum (int (len (pred [:, 4] >= metric_conf_threshold).sum ().item ()) for pred in predictions if pred.numel ())
    total_ground_truths = sum (int (len (targets ["labels"])) for targets in ground_truths)
    classes_with_gt = _classes_with_ground_truths (ground_truths, num_classes)
    
    return {
        "map50_95" : sum (ap_by_threshold.values ()) / max (1, len (ap_by_threshold)),
        "ap50" : ap_by_threshold.get (0.5, 0.0),
        "ap75" : ap_by_threshold.get (0.75, 0.0),
        "precision" : counts ["tp"] / max (1, counts ["tp"] + counts ["fp"]),
        "recall" : counts ["tp"] / max (1, counts ["tp"] + counts ["fn"]),
        "f1" : (2 * counts ["tp"]) / max (1, 2 * counts ["tp"] + counts ["fp"] + counts ["fn"]),
        "tp" : float (counts ["tp"]),
        "fp" : float (counts ["fp"]),
        "fn" : float (counts ["fn"]),
        "images" : float (len (ground_truths)),
        "ground_truths" : float (total_ground_truths),
        "predictions_at_conf" : float (total_predictions),
        "avg_predictions_per_image_at_conf" : total_predictions / max (1, len (predictions)),
        "classes_with_gt" : float (len (classes_with_gt)),
        "metric_conf_threshold" : float (metric_conf_threshold),
        "match_iou_threshold" : float (match_iou_threshold),
    }
    
def _classes_with_ground_truths (ground_truths, num_classes):
    classes = set ()
    
    for target in ground_truths:
        for label in target ["labels"].tolist ():
            label_int = int (label)
            
            if 0 <= label_int < num_classes:
                classes.add (label_int)
    
    return classes

def _mean_average_precision_at_iou (predictions, ground_truths, num_classes, iou_threshold):
    aps = []
    
    for class_id in range (num_classes):
        gt_by_iamge = {}
        gt_count = 0
        
        for image_index, targets in enumerate (ground_truths):
            mask = targets ["labels"] == class_id
            boxes = targets ["boxes"][mask].float ()
            
            if len (boxes):
                gt_by_iamge [image_index] = boxes
                gt_count += len (boxes)
                
        if gt_count == 0:
            continue
        
        class_predictions = []
        
        for image_index, pred in enumerate (predictions):
            if pred.numel () == 0:
                continue
            
            mask = pred [:, 5].long () == class_id
            
            for row in pred [mask]:
                class_predictions.append ((image_index, float (row [4]), row [:4].float ()))

            aps.append (_average_precision_for_class (class_predictions, gt_by_iamge, gt_count, iou_threshold))
            
        return sum (aps) / max (1, len (aps))
    
def _average_precision_for_class (class_predictions, gt_by_image, gt_count, iou_threshold):
    if not class_predictions:
        return 0.0
    
    matched = {image_index : torch.zeros (len (boxes), dtype = torch.bool) for image_index, boxes in gt_by_image.items ()}
    sorted_predictions = sorted (class_predictions, key = lambda item: item [1], reverse = True)
    tp = []
    fp = []
    
    for image_index, _score, box in sorted_predictions:
        gt_boxes = gt_by_image.get (image_index)
        
        if gt_boxes is None or len (gt_boxes) == 0:
            fp.append (1.0)
            tp.append (0.0)
            continue
        
        ious = box_iou (box.unsqueeze (0), gt_boxes).squeeze (0)
        best_iou, best_idx = ious.max (dim = 0)
        best_idx_int = int (best_idx.item ())
        
        if float (best_iou) >= iou_threshold and not bool (matched [image_index][best_idx_int]):
            tp.append (1.0)
            fp.append (0.0)
            matched [image_index][best_idx_int] = True
        else:
            fp.append (1.0)
            tp.append (0.0)
            
    tp_cum = torch.tensor (tp).cumsum (0)
    fp_cum = torch.tensor (fp).cumsum (0)
    recalls = tp_cum / max (1, gt_count)
    precisions = tp_cum / torch.clamp (tp_cum + fp_cum, min = 1e-12)
    
    return _coco_101_point_ap (precisions, recalls)

def _coco_101_point_ap (precisions, recalls):
    if precisions.numel () == 0 or recalls.numel () == 0:
        return 0.0
    
    recall_levels = torch.linspace (0, 1, 101)
    values = []
    
    for recall in recall_levels:
        eligible = precisions [recalls >= recall]
        values.append (float (eligible.max ()) if eligible.numel () else 0.0)
        
    return sum (values) / len (values)

def _precision_recall_counts (predictions, ground_truths, metric_conf_threshold, match_iou_threshold):
    matched = {}
    total_gt = 0
    
    for image_index, target in enumerate (ground_truths):
        total_gt += len (target ["labels"])
        
        for class_id in target ["labels"].unique ().tolist ():
            class_int = int (class_id)
            count = int ((target ["labels"] == class_int).sum ().item ())
            matched [(image_index, class_int)] = torch.zeros (count, dtype = torch.bool)
            
    candidates = []
    
    for image_index, pred in enumerate (predictions):
        if pred.numel () == 0:
            continue
        
        keep = pred [:, 4] >= metric_conf_threshold
        
        for row in pred [keep]:
            candidates.append ((image_index, int (row [5].item ()), float (row [4].item ()), row [:4].float ()))
            
    candidates.sort (key = lambda item: item [2], reverse = True)
    
    tp = 0
    fp = 0
    
    for image_index, class_id, _score, box in candidates:
        target = ground_truths [image_index]
        gt_mask = target ["labels"] == class_id
        gt_boxes = target ["boxes"] [gt_mask].float ()
        
        if len (gt_boxes) == 0:
            fp += 1
            continue
        
        ious = box_iou (box.unsqueeze (0), gt_boxes).squeeze (0)
        best_iou, best_idx = ious.max (dim = 0)
        key = (image_index, class_id)
        best_idx_int = int (best_idx.item ())
        
        if float (best_iou) >= match_iou_threshold and not bool (matched [key][best_idx_int]):
            matched [key][best_idx_int] = True
            tp += 1
        else:
            fp += 1
            
    return {"tp" : tp, "fp" : fp, "fn" : max (0, total_gt - tp)}


