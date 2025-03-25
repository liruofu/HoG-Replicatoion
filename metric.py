import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from tqdm import tqdm
from torch.utils.data import ConcatDataset, DataLoader

from dataloader import PositiveDataset, NegativeDataset, SlidingWindowDataset
from models import rHoG, SvmDetector


def det_curve_fppw(true_labels, pred_scores):
    miss_rates = []
    fppws = []
    threshold = np.linspace(np.min(pred_scores), np.max(pred_scores), 100)
    for t in threshold:
        pred_labels = (pred_scores >= t).astype(int)
        TP = np.sum((pred_labels == 1) & (true_labels == 1))
        FP = np.sum((pred_labels == 1) & (true_labels == 0))
        FN = np.sum((pred_labels == 0) & (true_labels == 1))
        miss_rate = FN / (TP + FN) if (TP + FN) > 0 else 0
        fppw = FP / len(pred_labels)
        miss_rates.append(miss_rate)
        fppws.append(fppw)

    return miss_rates, fppws


def plot_det_curves(names, fppw_list, miss_rate_list, loc='lower left'):
    plt.figure(figsize=(10, 6))

    # 定义颜色和符号
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'brown']
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'X', 'd']

    # 绘制每组 DET 曲线
    for i, (name, fppw, miss_rate) in enumerate(zip(names, fppw_list, miss_rate_list)):
        plt.plot(fppw, miss_rate, color=colors[i % len(colors)], marker=markers[i % len(markers)], label=name)

    # 设置双对数刻度
    plt.xscale('log')
    plt.yscale('log')

    # 添加标签和标题
    plt.xlabel('False Positives Per Window (FPPW)')
    plt.ylabel('Miss Rate')
    plt.title('Detection Error Tradeoff (DET) Curve')

    # 添加图例和网格
    plt.legend(loc=loc)
    # plt.grid(True, which="both", ls="--", linewidth=0.1)

    # 显示图像
    plt.show()


if __name__ == '__main__':
    names = ['Lin. R-HOG','b=4']
    positive_dataset = PositiveDataset("data/INRIAPerson/train_64x128_H96/pos")
    negative_dataset = NegativeDataset("data/INRIAPerson/train_64x128_H96/neg")
    train_dataset = ConcatDataset([positive_dataset, negative_dataset])
    data_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    for batch in data_loader:
        train_data = batch[0].numpy()
        train_label = batch[1].numpy()
    print("Train data shape:", train_data.shape)

    hog = rHoG()
    hog2 = rHoG(nbins=4)
    svm = SVC(kernel='linear', C=0.01)
    svm.fit(hog(train_data), train_label)
    svm1 = SVC(kernel='linear', C=0.01)
    svm1.fit(hog2(train_data), train_label)
    positive_dataset = PositiveDataset("data/INRIAPerson/test_64x128_H96/pos")
    negative_dataset = NegativeDataset("data/INRIAPerson/test_64x128_H96/neg", num_patches=20)
    test_dataset = ConcatDataset([positive_dataset, negative_dataset])
    data_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
    for batch in data_loader:
        test_data = batch[0].numpy()
        test_label = batch[1].numpy()
    scores = svm.decision_function(hog(test_data))
    miss_rates, fppws = det_curve_fppw(test_label, scores)
    m1, f1 = det_curve_fppw(test_label, svm1.decision_function(hog2(test_data)))
    plot_det_curves(names, [fppws,f1], [miss_rates,m1])
