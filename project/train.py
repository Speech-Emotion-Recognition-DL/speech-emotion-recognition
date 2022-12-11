import numpy as np
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, random_split

from myDataset import SoundDataset
from cnn import CNNNetwork
from project.cnn_model_definition import Convolutional_Speaker_Identification
from cnn_model import Convolutional_Neural_Network

"""
Batch size is a term used in machine learning and refers to the number of training examples utilized in one iteration"""
BATCH_SIZE = 128
EPOCHS = 30
# LEARNING_RATE = 0.001
LEARNING_RATE = 0.0001

ANNOTATIONS_FILE = '../project/Train_test_.csv'
SAMPLE_RATE = 16000
NUM_SAMPLES = 22050




def create_data_loader(data, batch_size):
    """
    Data loader is a class that we can use to wrap a data set in our case te train data
    and it will allow us to fetch data to load data in batches

    so it allow us to load data sets that are heavy on the memory without having any issues

    """

    train = int(data.__len__() * 0.8)
    test = int(data.__len__() * 0.2)
    train_data, test_data = random_split(dataset=data, lengths=[train, test])
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=1)
    return train_dataloader, test_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    """
    Loop through all the samples in the dataset and at each iteration we'll get a new batch of samples
    and this where the data loader comes in very handy because its an issue ball which will return us both the input and
    the target or in other word the x's and y's for one batch at each iteration`

    """

    for input, target in data_loader:
        optimiser.zero_grad()
        input, target = input.to(device), target.to(device)
        # print("target", target)

        # calculate loss
        prediction = model(input)

        # print(prediction)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights

        loss.backward()

        optimiser.step()

    print(f"loss: {loss.item()}")


def predict(model, test_loader):
    # model.eval()
    # with torch.no_grad():
    #     predictions = model(input)
    #     # Tensor (the number of sample that we are passing to the model = 1, the number of classes that the modle tries to predict = 10)
    #     # Tensor ( 1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
    #     # We are interested in the index that has the highest value.
    #     # So that index will correspond to the class that we want to predict.
    #     predicted_index = predictions[0].argmax(0)
    #     # if predicted_index == 8:
    #     #     predicted_index = 7
    #
    #     predicted = class_mapping[predicted_index]
    #     expected = class_mapping[target]

    model.eval()
    # x= audio
    # y= target
    pred_list = []
    target = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            y_pred = model(x_batch)
           # print(y_pred)
            # print(y_pred)
            # pred_list.append(y_pred[0].argmax())
            pred_list.append(np.argmax(y_pred.cpu().detach().numpy()))
            target.append(y_batch.cpu()[0])
            # print(pred_list)
            #print(target)

    accuracy = (np.array(pred_list) == np.array(target)).sum() / len(target) * 100


    return accuracy


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        model.train()
        print(f"Epoch {i + 1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    # instantiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    model_wev2vec = bundle.get_model().to(device)

    usd = SoundDataset(ANNOTATIONS_FILE,
                       model_wev2vec,
                       mel_spectrogram,
                       SAMPLE_RATE,
                       NUM_SAMPLES,
                       device)

    print(usd[1][0])
    #
    # train_dataloader, test_dataloader = create_data_loader(usd, BATCH_SIZE)
    #
    # # construct model and assign it to device
    # # cnn = CNNNetwork().to(device)
    # cnn = Convolutional_Neural_Network().to(device)
    # # cnn = Convolutional_Speaker_Identification().to(device)
    #
    # # print(cnn)
    #
    # # initialise loss funtion + optimiser
    # loss_fn = nn.CrossEntropyLoss()
    # optimiser = torch.optim.Adam(cnn.parameters(),
    #                              lr=LEARNING_RATE)
    #
    # # train model
    # train(cnn.cuda(), train_dataloader, loss_fn, optimiser, device, EPOCHS)
    #
    #
    #
    # acu = predict(cnn.cuda(), test_dataloader )
    # print(acu)
    # save model
    #torch.save(cnn.state_dict(), "feedforwardnet.pth")
    #print("Trained feed forward net saved at feedforwardnet.pth")
