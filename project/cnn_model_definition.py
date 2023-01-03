"""******************************************************************
The code is based on : https://github.com/a-nagrani/VGGVox/issues/1
******************************************************************"""

from torch import nn
import constants as c
import torch
from torchsummary import summary

DROP_OUT = 0.5
DIMENSION = 512 * 300


class Convolutional_Speaker_Identification(nn.Module):

    def cal_paddind_shape(self, new_shape, old_shape, kernel_size, stride_size):
        return (stride_size * (new_shape - 1) + kernel_size - old_shape) / 2

    def __init__(self):
        super().__init__()

        self.conv_2d_1 = nn.Conv2d(in_channels=1,
                                   out_channels=96,
                                   kernel_size=(7, 7),
                                   stride=(2, 2),
                                   padding=1)
        self.bn_1 = nn.BatchNorm2d(96)
        self.max_pool_2d_1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.conv_2d_2 = nn.Conv2d(96, 256, kernel_size=(5, 5), stride=(2, 2), padding=1)
        self.bn_2 = nn.BatchNorm2d(256)
        self.max_pool_2d_2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.conv_2d_3 = nn.Conv2d(256, 384, kernel_size=(3, 3), padding=1)
        self.bn_3 = nn.BatchNorm2d(384)

        self.conv_2d_4 = nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1)
        self.bn_4 = nn.BatchNorm2d(256)

        self.conv_2d_5 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.bn_5 = nn.BatchNorm2d(256)
        self.max_pool_2d_3 = nn.MaxPool2d(kernel_size=(5, 3), stride=(3, 2))

        self.conv_2d_6 = nn.Conv2d(256, 4096, kernel_size=(2, 1), padding=0)
        self.drop_1 = nn.Dropout(p=DROP_OUT)

        self.global_avg_pooling_2d = nn.AdaptiveAvgPool2d((1, 1))
        self.dense_1 = nn.Linear(4096, 1024)
        self.drop_2 = nn.Dropout(p=DROP_OUT)

        self.dense_2 = nn.Linear(1024, 3)

    def forward(self, X):
        # print(X.shape)
        x = nn.ReLU()(self.conv_2d_1(X))
        # print(x.shape)
        x = self.bn_1(x)

        x = self.max_pool_2d_1(x)

        x = nn.ReLU()(self.conv_2d_2(x))
        # print(x.shape)
        x = self.bn_2(x)
        # print(x.shape)
        x = self.max_pool_2d_2(x)

        x = nn.ReLU()(self.conv_2d_3(x))

        x = self.bn_3(x)

        x = nn.ReLU()(self.conv_2d_4(x))

        x = self.bn_4(x)

        x = nn.ReLU()(self.conv_2d_5(x))

        x = self.bn_5(x)

        x = self.max_pool_2d_3(x)

        x = nn.ReLU()(self.conv_2d_6(x))

        x = self.drop_1(x)

        x = self.global_avg_pooling_2d(x)

        x = x.view(-1, x.shape[1])  # output channel for flatten before entering the dense layer

        x = nn.ReLU()(self.dense_1(x))

        x = self.drop_2(x)

        x = self.dense_2(x)

        y = nn.LogSoftmax(dim=1)(x)  # consider using Log-Softmax

        return x, y

    def get_epochs(self):
        return 3

    def get_learning_rate(self):
        return 0.0001

    def get_batch_size(self):
        return 16

    def to_string(self):
        return "Convolutional_Speaker_Identification_Log_Softmax_Model-epoch_"


if __name__ == "__main__":
    cnn = Convolutional_Speaker_Identification()
    # summary(cnn, (1, 64, 44))
    summary(cnn.cuda(), (1, 149, 768))

"""
├─Conv2d: 1-1                            [-1, 96, 73, 382]         4,800
├─BatchNorm2d: 1-2                       [-1, 96, 73, 382]         192
├─MaxPool2d: 1-3                         [-1, 96, 36, 190]         --
├─Conv2d: 1-4                            [-1, 256, 17, 94]         614,656
├─BatchNorm2d: 1-5                       [-1, 256, 17, 94]         512
├─MaxPool2d: 1-6                         [-1, 256, 8, 46]          --
├─Conv2d: 1-7                            [-1, 384, 8, 46]          885,120
├─BatchNorm2d: 1-8                       [-1, 384, 8, 46]          768
├─Conv2d: 1-9                            [-1, 256, 8, 46]          884,992
├─BatchNorm2d: 1-10                      [-1, 256, 8, 46]          512
├─Conv2d: 1-11                           [-1, 256, 8, 46]          590,080
├─BatchNorm2d: 1-12                      [-1, 256, 8, 46]          512
├─MaxPool2d: 1-13                        [-1, 256, 2, 22]          --
├─Conv2d: 1-14                           [-1, 4096, 2, 22]         1,052,672
├─Dropout: 1-15                          [-1, 4096, 2, 22]         --
├─AdaptiveAvgPool2d: 1-16                [-1, 4096, 1, 1]          --
├─Linear: 1-17                           [-1, 1024]                4,195,328
├─Dropout: 1-18                          [-1, 1024]                --
├─Linear: 1-19                           [-1, 3]      
"""
