import torch
from collections import Counter
from IOU import IOU

# Đinhh nghĩa hàm mAp
def mAp(pred_boxes, true_boxes, iou_threshold=0.5, box_format="corners", num_classes=20):
    # pred_boxes = [[train_idx, pred_class, prob_score, x1, y1, x2, y2], [], []]
    # true_boxes = [[train_idx, true_class, truth_score, x1, y1, x2, y2], [], []]

    
    # define average_precision
    average_precisions = []
    epsilon = 1e-6
    
    # Giả sử cần phân loại chó mèo với chó: true_class = 0 và mèo: true_class = 1
    # Có 3 bức ảnh với idx lần lượt là 0, 1, 2
    
    
    # Duyệt qua từng class
    for c in range(num_classes):
        # Xác định pred_boxes cho từng class
        detections = []
        
        # Xác định true_boxes cho từng class
        ground_truths = []
        
        # Lấy từng pred box tương ứng vỡi class đang xét
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)
                
        # Lấy từng true box tương ứng với class đang xét
        for true_box in true_box:
            if true_box[1] == c:
                ground_truths.append(true_box)
        
        
        # Đếm số true box có trong image
        # image 1 có idx = 1 và có 3 true box
        # image 2 có idx = 2 và có 5 true box
        # amount_bboxes = ({1 : 3, 2 : 5})
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        
        # Duyệt qua từng phần tử trong amount_bboxes
        # amount_bboxes = ({1: (0, 0, 0), 2:(0, 0, 0, 0, 0)})
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)
            
        # Xắp xếp các predict bouding box theo thứ tự giảm dần từ probability score
        detections.sort(key=lambda x : x[2], reverse=True)
        
        # Định nghĩa dương tính thật: True Positive - TP
        # Giả sử tương ứng với mỗi idx có 7 detection bouding box
        # TP = (0, 0, 0, 0, 0, 0, 0, 0)
        TP = torch.zeros((len(detections)))
        
        # Định nghĩa âm tính thật: False Positive - FP
        # FP = (0, 0, 0, 0, 0, 0, 0)
        FP = torch.zeros((len(detections)))
        
        # Định nghĩa tổng số bouding box trong ground_truths
        total_true_bboxes = len(ground_truths)
        
        
        for detection_idx, detection in enumerate(detections):
            # Lấy ra tất cả các true box trong cùng image với pred box
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]
            
            # Tính số lượng các true box trong image
            num_gts = len(ground_truth_img)
            
            # Define best iou
            best_iou = 0
            
            # Duyệt qua từng bouding box đúng
            for idx, gt in enumerate(ground_truth_img):
                # Tính iou của pred_box tương ứng với từng true box trong cùng một picture
                iou = IOU(torch.tensor(detection[3:]),
                          torch.tensor(gt[3:]),
                          box_format=box_format)

                # Kiêm tra iou
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else :
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1
        
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1])), precisions)
        recalls = torch.cat((torch.tensor([1])), recalls)
        
        average_precisions.append(torch.trapz(precisions, recalls))
    
    return sum(average_precisions) / len(average_precisions)
        