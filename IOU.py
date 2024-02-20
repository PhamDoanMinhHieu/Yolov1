import torch
# Định nghĩa hàm tính intersection over union
def IOU(boxes_preds, boxes_labels, box_format="midpoint"):
    # boxes_preds shape is (N, 4) where N is a number of boxes
    # boxes_labels shape is (N, 4) where N is a number of boxes
    # box_format: midpoint or corners
    # midpoint: (x_center, y_center, width, heigth)
    # corners: (x1, y1, x2, y2)    
    
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2    
        
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
        
    elif box_format == "corners":
        # take point
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]
    
    # Find coordinate of IOU
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    
    # clamp(0) is for the case when they do not intersection
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    
    # area of box1
    box1_area = abs(box1_x1 - box1_x2) * abs(box1_y1 - box1_y2)
    box2_area = abs(box2_x1 - box2_x2) * abs(box2_y1 - box2_y2)
    
    # return intersection
    return intersection / (box1_area + box2_area - intersection + 1e-6)
    
    
