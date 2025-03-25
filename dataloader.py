import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage import io
from skimage.transform import resize
import numpy as np
import cv2
import os
from glob import glob


class PositiveDataset(Dataset):
    def __init__(self, img_dir, img_size=(128, 64)):
        self.image_paths = sorted(glob(os.path.join(img_dir, "*.png")))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(img_size),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = io.imread(self.image_paths[idx])
        img = np.asarray(self.transform(img), dtype=np.uint8)
        return torch.tensor(img[:, :, :3]).permute(2, 0, 1), 1


class NegativeDataset(Dataset):
    def __init__(self, init_neg_dir, num_patches=10, img_size=(128, 64)):
        self.init_neg_images = sorted(glob(os.path.join(init_neg_dir, "*.png")))
        self.num_patches = num_patches
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(img_size),
        ])

    def __len__(self):
        return len(self.init_neg_images) * self.num_patches

    def __getitem__(self, idx):
        img_idx = idx // self.num_patches
        img_path = self.init_neg_images[img_idx]
        img = io.imread(img_path)
        patch = self.transform(img)
        patch = np.asarray(patch, dtype=np.uint8)
        return torch.tensor(patch).permute(2, 0, 1), 0  # 负样本标签为0


def parse_pascal_annotation(ann_file):
    boxes = []
    with open(ann_file, 'r', encoding='latin-1') as f:
        lines = f.readlines()
        for line in lines:
            if "Bounding box for object" in line:
                parts = line.split(':')[-1].strip().split(' - ')
                xmin_ymin = parts[0].strip('()').split(', ')
                xmax_ymax = parts[1].strip('()').split(', ')
                xmin, ymin = int(xmin_ymin[0]), int(xmin_ymin[1])
                xmax, ymax = int(xmax_ymax[0]), int(xmax_ymax[1])
                boxes.append([xmin, ymin, xmax, ymax])
    return boxes


class SlidingWindowDataset:
    def __init__(self, root_dir, window_size=(128, 64), step_size=8, pos=True, scale_factor=1.2):
        self.root_dir = root_dir
        self.window_size = window_size
        self.step_size = step_size
        self.scale_factor = scale_factor
        self.pos = pos
        self.annotation_dir = os.path.join(root_dir, "Test/annotations")
        if pos is True:
            self.dir = os.path.join(root_dir, "Test/pos")
        else:
            self.dir = os.path.join(root_dir, "Test/neg")
        self.images = sorted([os.path.join(self.dir, f) for f in os.listdir(self.dir)
                              if f.endswith(('.jpg', '.png'))])
        if pos is True:
            self.annotations = {}
            for img_path in self.images:
                ann_file = os.path.join(self.annotation_dir,
                                        os.path.basename(img_path).replace('.png', '.txt').replace('.jpg', '.txt'))
                if os.path.exists(ann_file):
                    boxes = parse_pascal_annotation(ann_file)
                    self.annotations[img_path] = boxes

    def sliding_window(self, image):
        """滑动窗口"""
        for y in range(0, image.shape[0] - self.window_size[0] + 1, self.step_size):
            for x in range(0, image.shape[1] - self.window_size[1] + 1, self.step_size):
                window = image[y:y + self.window_size[0], x:x + self.window_size[1]]
                if window.shape[0] != self.window_size[0] or window.shape[1] != self.window_size[1]:
                    continue
                yield x, y, window

    def compute_iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x1_gt, y1_gt, x2_gt, y2_gt = box2
        xi1, yi1 = max(x1, x1_gt), max(y1, y1_gt)
        xi2, yi2 = min(x2, x2_gt), min(y2, y2_gt)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area

    def assign_labels(self, positions, scales, gt_boxes):
        labels = np.zeros(len(positions), dtype=int)
        if not gt_boxes:
            return labels

        for i, (pos, scale) in enumerate(zip(positions, scales)):
            x, y = pos
            h, w = self.window_size
            det_box = [x * scale, y * scale, (x + w) * scale, (y + h) * scale]
            for gt in gt_boxes:
                iou = self.compute_iou(det_box, gt)
                if iou >= 0.5:
                    labels[i] = 1
                    break
        return labels

    def __len__(self):
        return len(self.images)

    def image_pyramid(self, image):
        yield image
        while True:
            w = int(image.shape[1] / self.scale_factor)
            h = int(image.shape[0] / self.scale_factor)
            image = resize(image, (h, w, 3),
                           preserve_range=True, anti_aliasing=False).astype(np.uint8)
            if h < self.window_size[0] or w < self.window_size[1]:
                break
            yield image

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = io.imread(img_path)[:, :, :3]
        # 滑动窗口生成
        windows = []
        positions = []
        scales = []
        for scaled_image in self.image_pyramid(image):
            scale = image.shape[0] / scaled_image.shape[0]
            for (x, y, window) in self.sliding_window(scaled_image):
                windows.append(np.transpose(window, (2, 0, 1)))
                positions.append((x, y))
                scales.append(scale)
        if self.pos:
            gt_boxes = self.annotations.get(img_path, [])
            labels = self.assign_labels(positions, scales, gt_boxes)
        else:
            labels = np.zeros(len(windows), dtype=int)
        return np.array(windows, dtype=np.uint8), labels


# 使用示例
# 测试代码
if __name__ == "__main__":
    data_set = SlidingWindowDataset('data/INRIAPerson')
    for data in data_set:
        windows, labels = data
        print(windows.shape)
