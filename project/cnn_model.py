"""******************************************************************
The code is based on : https://github.com/a-nagrani/VGGVox/issues/1
******************************************************************"""

from torch import nn
import torch
from torchsummary import summary

DROP_OUT = 0.5
DIMENSION = 512 * 300


class Convolutional_Neural_Network(nn.Module):

    def cal_paddind_shape(self, new_shape, old_shape, kernel_size, stride_size):
        return (stride_size * (new_shape - 1) + kernel_size - old_shape) / 2

    def __init__(self):
        super().__init__()
        self.conv_2d_1 = nn.Conv2d(
            in_channels=1,  # input volume depth == input channel dim == 1
            out_channels=16,  # expand output feature map volume's depth to 16
            kernel_size=(7, 7),  # typical 3*3 stride 1 kernel
            stride=(2, 2),
            padding=1)  # starting stride = (2, 2)
        self.bn_1 = nn.BatchNorm2d(16)  # batch normalize the output feature map before activation
        self.max_pool_2d_1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.conv_2d_2 = nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(2, 2), padding=1)  # starting stride = (2, 2)
        self.bn_2 = nn.BatchNorm2d(32)
        self.max_pool_2d_2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.conv_2d_3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn_3 = nn.BatchNorm2d(64)

        self.conv_2d_4 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn_4 = nn.BatchNorm2d(128)

        self.conv_2d_5 = nn.Conv2d(32, 128, kernel_size=(3, 3), padding=1)
        self.bn_5 = nn.BatchNorm2d(128)
        self.max_pool_2d_3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        # self.conv_2d_6 = nn.Conv2d(128, 256, kernel_size=(3, 1), padding=0)
        self.conv_2d_6 = nn.Conv2d(128, 256, kernel_size=(2, 1), padding=0)
        self.drop_1 = nn.Dropout(p=DROP_OUT)

        # self.global_avg_pooling_2d = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        # self.dense_1 = nn.Linear(35328, 1024)
        # self.dense_1 = nn.Linear(11776, 1024)
        self.dense_1 = nn.Linear(8192, 1024)
        self.drop_2 = nn.Dropout(p=DROP_OUT)

        self.dense_2 = nn.Linear(1024, 3)

    def forward(self, X):
        # print(X.shape)  # batch_size, channels, height, width = x.shape
        # print("conv 1")
        x = nn.ReLU()(self.conv_2d_1(X))
        # print(x.shape)
        x = self.bn_1(x)
        # print(x.shape)
        # print("conv 2")
        x = nn.ReLU()(self.conv_2d_2(x))
        # print(x.shape)
        x = self.bn_2(x)
        # print(x.shape)
        x = self.max_pool_2d_1(x)
        # print(x.shape)

        # x = self.max_pool_2d_2(x)

        # print(x.shape)

        # x = nn.ReLU()(self.conv_2d_3(x))
        # x = self.bn_3(x)

        # # print(x.shape)

        # x = nn.ReLU()(self.conv_2d_4(x))
        # x = self.bn_4(x)

        # print("po ",x.shape)

        x = nn.ReLU()(self.conv_2d_5(x))
        # print("po1 ", x.shape)
        x = self.bn_5(x)

        # print(x.shape)

        x = nn.ReLU()(self.conv_2d_6(x))
        # print("po 3 " , x.shape)
        x = self.drop_1(x)
        # x = self.global_avg_pooling_2d(x)
        # print(x.shape)
        # x = self.max_pool_2d_3(x)
        # x = x.view(-1, x.shape[1])  # output channel for flatten before entering the dense layer
        x = self.flatten(x)

        # print("flatt" ,x.shape)

        x = nn.ReLU()(self.dense_1(x))
        # print("relu ",x.shape)
        x = self.drop_2(x)

        # print(x.shape)

        x = self.dense_2(x)

        # print(x.shape)

        # y = torch.sigmoid(x)
        activation = nn.Softmax(dim=1)
        y = activation(x)

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
    cnn = Convolutional_Neural_Network()
    # summary(cnn, (1, 64, 44))

    summary(cnn.cuda(), (1, 149, 29))  # the shape of the signal
    # summary(cnn.cuda(), (1, 68, 29
