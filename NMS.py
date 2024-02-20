import torch
from IOU import IOU

# Định nghĩa hàm Non Max Suppression
def nms(bboxes, iou_threshold, prob_threshold, box_format="corners"):
    # predictions = [[1, 0.9, x1, y1, x2, y2], [], []]
    
    assert type(bboxes) == list
    
    bboxes = [box for box in bboxes if box[1] > prob_threshold]
    bboxes_after_nms = []
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
       
    while bboxes:
        chosen_box = bboxes.pop(0)
        
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or IOU(torch.tensor(chosen_box[2:]), torch.tensor(box[2:], box_format=box_format)) < iou_threshold
        ]
        
        bboxes_after_nms.append(chosen_box)
    
    return bboxes_after_nms