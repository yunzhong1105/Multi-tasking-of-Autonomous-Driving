import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

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
        # print(np.array(x))
        x = self.fc(x)
        return x
    


    
class ClsHead(nn.Module):
    def __init__(self, num_classes):
        super(ClsHead, self).__init__()
        # Global Average Pooling layers
        self.gap_1 = nn.AdaptiveAvgPool2d((1, 1))
        self.gap_2 = nn.AdaptiveAvgPool2d((1, 1))
        self.gap_3 = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected layers
        self.fc_1 = nn.Linear(768, 256)  # 768 is the last dimension of the third feature shape
        self.fc_2 = nn.Linear(384, 256)  # 384 is the last dimension of the second feature shape
        self.fc_3 = nn.Linear(192, 256)  # 192 is the last dimension of the first feature shape
        
        self.classifier = nn.Linear(256 * 3, num_classes)  # Concatenate outputs of all FC layers
        
    def forward(self, x1, x2, x3):
        x1 = self.gap_1(x1)
        x1 = torch.flatten(x1, 1)
        x1 = self.fc_1(x1)
        x1 = F.relu(x1)
        
        x2 = self.gap_2(x2)
        x2 = torch.flatten(x2, 1)
        x2 = self.fc_2(x2)
        x2 = F.relu(x2)
        
        x3 = self.gap_3(x3)
        x3 = torch.flatten(x3, 1)
        x3 = self.fc_3(x3)
        x3 = F.relu(x3)
        
        # Concatenate the outputs of all Fully Connected layers
        x = torch.cat((x1, x2, x3), dim=1)
        
        # Classification layer
        x = self.classifier(x)
        return x

# Example usage:
# num_classes = 10  # Change this according to your classification task
# model_head = ClsHead(num_classes)
# x1 = torch.randn(8, 192, 40, 40)
# x2 = torch.randn(8, 384, 20, 20)
# x3 = torch.randn(8, 768, 10, 10)
# output = model_head(x1, x2, x3)
# print(output.shape)  # Output shape will be [batch_size, num_classes]




class ClsConv(nn.Module):
    def __init__(self, in_channels = 192 , num_classes = 196 , L_channels = 64 , make_prediction = False) :
        super(ClsConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels = 64 , kernel_size = 3 , stride = 1 , padding = 1)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = (3 , 3))
        # self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.make_prediction = make_prediction
        if self.make_prediction :
            self.fc = nn.Linear(L_channels , num_classes)
            self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        if not self.make_prediction :
            x = self.conv(x)
            x = self.relu(x)
            x = self.maxpool(x)
        else :
            x = torch.flatten(x , 1)  
            x = self.fc(x)
            x = self.softmax(x)
        return x

# # 创建模型头
# head_1 = ModelHead(192)
# head_2 = ModelHead(384)
# head_3 = ModelHead(768)

# # 分别将不同形状的特征输入到模型头中
# output_1 = head_1(feature_1)
# output_2 = head_2(feature_2)
# output_3 = head_3(feature_3)

