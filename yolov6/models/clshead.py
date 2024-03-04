import torch.nn as nn
import numpy as np

class ClassificationHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 512),  # 假設這裡使用一個包含 512 個神經元的隱藏層
            nn.ReLU(),
            nn.Dropout(0.5),  # 可選的 dropout 層
            nn.Linear(512, num_classes)  # 輸出層，數量等於分類的類別數
        )

    def forward(self, x):
        print(np.array(x))
        x = self.fc(x)
        return x