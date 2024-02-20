"""
Main file for training Yolo model on Pascal VOC dataset

"""

import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from loss import YoloLoss

seed = 123
torch.manual_seed(seed)

# Các siêu tham số
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 16 
WEIGHT_DECAY = 0
EPOCHS = 1000
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes

# Resize bức ảnh về (448, 448) và chuyển labels thành các tensor
transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])


def train_fn(train_loader, model, optimizer, loss_fn):
    # train_loader được sử dụng để tải dữ liệu huấn luyện, tăng tốc độ dữ liệu và quản lí bộ nhớ
    # tqdm(train_loader, leave=True) được sử dụng để theo dõi tiến trình huấn luyện, leave = True để lưu lại tiến trình sau mỗi epoch
    loop = tqdm(train_loader, leave=True)
    
    # mean average để lưu trữ giá trị của hàm loss trong suốt quá trình huấn luyện
    mean_loss = []

    # Bắt đầu vòng lặp qua các batch trong train_loader
    # Mỗi batch sẽ có dữ liệu đầu vào x và nhãn y
    for batch_idx, (x, y) in enumerate(loop):
        
        # Chuyển dữ liệu đầu vào x và nhãn y sang thiết bị DEVICE để tối ưu tính toán
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        # Đưa dữ liệu x vào mô hình và dự đoán đầu ra out
        out = model(x)
        
        # Tính toán hàm mất mát bằng cách so sánh nhãn và giá trị dự đoán
        loss = loss_fn(out, y)
        
        # Thêm giá trị mất mát hiện tại vào danh sách để tính giá trị loss trung bình sau khi hoàn tất quá trình huấn luyện
        mean_loss.append(loss.item())
        
        # Đặt gradient của tất cả các tham số về 0 trước khi tính toán gradient mới
        optimizer.zero_grad()
        
        # Tính toán gradient của hàm mất mát theo các tham số trong mô hình
        loss.backward()
        
        # Tiến hành cập nhập tham số mô hình
        optimizer.step()

        # Cập nhập thông tin trong thanh tiến trình để hiển thị giá trị mất mát hiện tại
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")


def main():
    # Khởi tạo kiến trúc Yolov và chuyển sang thiết bị DEVICE để tối ưu tính toán
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    
    # Khởi tạo hàm tối ưu Adam với tham số cần huyến luyện là từ Yolov1
    optimizer = optim.Adam(
        model.parameters(), # Lấy tất cả các tham số của mô hình để cập nhập
        lr=LEARNING_RATE,   # Đặt tốc độ học 
        weight_decay=WEIGHT_DECAY # Đặt hệ số suy giảm trọng lượng
    )
    
    # Khởi tạo hàm Loss
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    # Thực hiện lấy dữ liệu để train từ data
    train_dataset = VOCDataset(
        "data/100examples.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )
    
    # Chuyển dữ liệu để train thành DataLoader thuận tiện cho việc chia batch và tính toán
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    # Thực hiện lấy dữ liệu để test từ data
    test_dataset = VOCDataset(
        "data/test.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    # Chuyển dữ liệu để test thành DataLoader thuận tiện cho việc chia batch và tính toán
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    #Duyệt qua từng epoch
    for epoch in range(EPOCHS):
        
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec}")

        train_fn(train_loader, model, optimizer, loss_fn)


if __name__ == "__main__":
    main()