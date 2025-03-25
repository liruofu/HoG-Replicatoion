import numpy as np
import torch
import cv2
from skimage.transform import warp_polar
from torch.utils.data import ConcatDataset, DataLoader
from sklearn.svm import SVC, LinearSVC
from dataloader import PositiveDataset, NegativeDataset


class rHoG:
    def __init__(self, cell_size=(8, 8), block_size=(2, 2), stride=(8, 8),
                 nbins=9, signed_gradient=False, win_size=(64, 128)):
        block_size_pixels = (cell_size[0] * block_size[0], cell_size[1] * block_size[1])
        self.hog = cv2.HOGDescriptor(
            _winSize=win_size,  # 输入图像大小，需与数据匹配
            _blockSize=block_size_pixels,  # 例如 (16, 16)
            _blockStride=stride,  # 步幅通常等于 cell_size
            _cellSize=cell_size,
            _winSigma=0,
            _signedGradient=signed_gradient,  # 例如 (8, 8)
            _nbins=nbins  # 方向 bin 数量
        )

    def __call__(self, data):
        rhog_features = []

        for pic in data:
            features = []
            for c in pic:
                f = self.hog.compute(c)
                features.append(f)
            features = np.max(np.array(features), axis=0)
            rhog_features.append(features)
        return np.array(rhog_features)


def hard_example_mining_svm(svm, x_train, y_train, max_iter=2):
    for i in range(max_iter):
        svm.fit(x_train, y_train)
        y_pred = svm.predict(x_train)
        fp_indices = np.where((y_train == 0) & (y_pred == 1))[0]
        fp_count = len(fp_indices)
        if fp_count == 0:
            break
        x_train = np.concatenate((x_train, x_train[fp_indices]) ,axis=0)
        y_train = np.concatenate((y_train, y_train[fp_indices]))
    return svm


class SvmDetector:
    def __init__(self, processor, svm, train_data, labels):
        self.processor = processor
        self.svm = svm
        self.svm.fit(processor(train_data), labels)

    def __call__(self, data):
        return self.svm.decision_function(self.processor(data))


if __name__ == '__main__':
    positive_dataset = PositiveDataset("data/INRIAPerson/train_64x128_H96/pos")
    negative_dataset = NegativeDataset("data/INRIAPerson/train_64x128_H96/neg")
    train_dataset = ConcatDataset([positive_dataset, negative_dataset])
    data_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    for batch in data_loader:
        train_data = batch[0].numpy()
        train_label = batch[1].numpy()
    print("Train data shape:", train_data.shape)

    testHoG = rHoG()
    svm = LinearSVC(C=0.01)
    output = testHoG(train_data)
    svm = hard_example_mining_svm(svm, output, train_label)

    positive_dataset = PositiveDataset("data/INRIAPerson/test_64x128_H96/pos")

    data_loader = DataLoader(positive_dataset, batch_size=len(positive_dataset), shuffle=True)
    for batch in data_loader:
        test_data = batch[0].numpy()
        test_label = batch[1].numpy()

    output = testHoG(test_data)

    scores = svm.decision_function(output)
    print(np.max(scores))
