from torch import nn
from torchsummary import summary


class CNNNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        # 4 conv blocks / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()

        #self.linear = nn.Linear(128 * 5 * 4, 10)
        self.linear = nn.Linear(2560,10)
        #self.linear = nn.Linear(2304, 8)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        print("normal  --> ", input_data.size())
        x = self.conv1(input_data)
        # print("conv1  --> ", x.size())
        x = self.conv2(x)
        # print("conv2  --> ", x.size())
        x = self.conv3(x)
        # print("conv3  --> ", x.size())
        x = self.conv4(x)
        # print("conv4  --> ", x.size())
        x = self.flatten(x)
        # print("flatten  --> ", x.size())
        logits = self.linear(x)
        # print("linear  --> ", logits.size())
        predictions = self.softmax(logits)
        # print("predi  --> ", predictions)
        return predictions


if __name__ == "__main__":
    cnn = CNNNetwork()
    # summary(cnn, (1, 64, 44))

    summary(cnn.cuda(), (1, 68, 29))  # the shape of the signal
    # summary(cnn.cuda(), (1, 68, 29))
