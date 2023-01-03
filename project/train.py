import numpy as np
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, random_split
from myDataset import SoundDataset
from cnn import CNNNetwork
from project.cnn_model_definition import Convolutional_Speaker_Identification
from cnn_model import Convolutional_Neural_Network
from torchsummary import summary
from audiomentations import Compose, AddGaussianNoise, TimeMask, PitchShift, BandStopFilter
import matplotlib as plt
from cnn_b import ConvNet

"""
Batch size is a term used in machine learning and refers to the number of training examples utilized in one iteration"""
BATCH_SIZE = 128
EPOCHS = 20
# LEARNING_RATE = 0.001
LEARNING_RATE = 0.0003
# bundle = torchaudio.pipelines.WAV2VEC2_BASE
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
ANNOTATIONS_FILE = 'Train_test_.csv'
SAMPLE_RATE = bundle.sample_rate
NUM_SAMPLES = bundle.sample_rate
model = bundle.get_model()
augmentations = True
import random


def create_data_loader(data, batch_size):
    """
    Data loader is a class that we can use to wrap a data set in our case te train data
    and it will allow us to fetch data to load data in batches

    so it allow us to load data sets that are heavy on the memory without having any issues

    """

    train = int(data.__len__() * 0.8)
    test = data.__len__() - train
    # test = int(data.__len__() * 0.2)

    print("train: ", train)
    print("test: ", test)

    train_data, test_data = random_split(dataset=data, lengths=[train, test])
    # print("train data: ", train_data, test_data)

    train_dataloader = DataLoader(train_data, batch_size=batch_size)

    test_dataloader = DataLoader(test_data, batch_size=1)
    return train_dataloader, test_dataloader


augment1 = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.03, p=1),
    TimeMask(min_band_part=0.0, max_band_part=0.15, p=0),  # masks too much time in many cases
    PitchShift(min_semitones=-6, max_semitones=8, p=0),
    BandStopFilter(min_center_freq=60, max_center_freq=2500, min_bandwidth_fraction=0.1, max_bandwidth_fraction=0.4,
                   p=0)
])


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    """
    Loop through all the samples in the dataset and at each iteration we'll get a new batch of samples
    and this where the data loader comes in very handy because its an issue ball which will return us both the input and
    the target or in other word the x's and y's for one batch at each iteration`

    """

    for input, target in data_loader:
        num = random.randint(0, 2)

        # if num % 2 == 0:
        #     # print(input.shape)
        #     input = input.cpu()
        #     input = augment1(input, SAMPLE_RATE)
        optimiser.zero_grad()

        # Apply the transformations to the audio data
        # print(input.shape)

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
            # print(target)

    accuracy = (np.array(pred_list) == np.array(target)).sum() / len(target) * 100

    return accuracy


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        model.train()
        print(f"Epoch {i + 1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")


def validate(model, test_dataloader, device):
    """
    The function then iterates through the test dataloader
     and uses the model to make predictions on each batch of test data.
    It compares the predicted labels to the true labels and increments the correct counter if the prediction was correct.
     The function also keeps track of the total number of samples in the test set with the total variable.

    After iterating through all the test data, the function
    returns the accuracy of the model as a percentage, which is calculated by dividing the number
     of correct predictions by the total number of samples in the test set.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


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
    model_wav2vec = bundle.get_model().to(device)

    usd = SoundDataset(ANNOTATIONS_FILE,
                       model_wav2vec,
                       mel_spectrogram,
                       SAMPLE_RATE,
                       NUM_SAMPLES,
                       device, True)

    # print(usd[1][0])

    train_dataloader, test_dataloader = create_data_loader(usd, BATCH_SIZE)

    # construct model and assign it to device
    cnn = Convolutional_Speaker_Identification().to(device)
    #cnn = Convolutional_Neural_Network().to(device)
    ##cnn = ConvNet().to(device)
    # cnn = Convolutional_Speaker_Identification().to(device)
    # summary(cnn.cuda(), (1, 49, 768))  # the shape of the signal
    # print(cnn)

    # initialise loss funtion + optimiser

    """
    PyTorch nn.CrossEntropyLoss() implements log softmax and negative log likelihood loss 
    (nn.NLLoss() --> nn.LogSoftmax()) We use log softmax for computation benefits and faster
    gradient optimization. 
    Log softmax heavily penalizes the model when failing to predict the correct class."""
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(cnn.cuda(), train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # Save the model
    torch.save(cnn.cuda().state_dict(), 'model.pt')

    # Save the dataset and dataloader
    torch.save(test_dataloader.dataset, 'test_dataset.pt')
    torch.save(test_dataloader, 'test_dataloader.pt')

    torch.save(test_dataloader.dataset, 'train_dataset.pt')
    torch.save(test_dataloader, 'train_dataloader.pt')


    val = validate(cnn.cuda(), train_dataloader, device)
    print(val)

    acu = predict(cnn.cuda(), test_dataloader)
    print(acu)


    #
    # epochs = list(range(200))
    # acc = history.history['accuracy']
    # val_acc = history.history['val_accuracy']
    #
    # plt.plot(epochs, acc, label='train accuracy')
    # plt.plot(epochs, val_acc, label='val accuracy')
    # plt.xlabel('epochs')
    # plt.ylabel('accuracy')
    # plt.legend()
    # plt.show()


    ## without augmntion
    ##Test accuracy is 65.33%
    """
    Epoch 0: iteration 34/35
Epoch 0 --- loss:1.964, Epoch accuracy:22.49%, Validation loss:1.788, Validation accuracy:36.36%
Epoch 1: iteration 34/35
Epoch 1 --- loss:1.395, Epoch accuracy:44.38%, Validation loss:1.527, Validation accuracy:46.85%
Epoch 2: iteration 34/35
Epoch 2 --- loss:1.080, Epoch accuracy:58.06%, Validation loss:1.424, Validation accuracy:45.45%
Epoch 3: iteration 34/35
Epoch 3 --- loss:0.708, Epoch accuracy:72.62%, Validation loss:1.740, Validation accuracy:41.96%
Epoch 4: iteration 34/35
Epoch 4 --- loss:0.501, Epoch accuracy:80.91%, Validation loss:2.099, Validation accuracy:42.66%
Epoch 5: iteration 34/35
Epoch 5 --- loss:0.329, Epoch accuracy:86.66%, Validation loss:1.569, Validation accuracy:53.15%
Epoch 6: iteration 34/35
Epoch 6 --- loss:0.203, Epoch accuracy:90.58%, Validation loss:1.338, Validation accuracy:53.15%
Epoch 7: iteration 34/35
Epoch 7 --- loss:0.130, Epoch accuracy:94.07%, Validation loss:1.469, Validation accuracy:52.45%
Epoch 8: iteration 34/35
Epoch 8 --- loss:0.107, Epoch accuracy:94.86%, Validation loss:1.643, Validation accuracy:55.24%
Epoch 9: iteration 34/35
Epoch 9 --- loss:0.064, Epoch accuracy:95.82%, Validation loss:1.519, Validation accuracy:59.44%
Epoch 10: iteration 34/35
Epoch 10 --- loss:0.075, Epoch accuracy:95.64%, Validation loss:2.098, Validation accuracy:48.25%
Epoch 11: iteration 34/35
Epoch 11 --- loss:0.089, Epoch accuracy:94.86%, Validation loss:2.076, Validation accuracy:52.45%
Epoch 12: iteration 34/35
Epoch 12 --- loss:0.050, Epoch accuracy:95.99%, Validation loss:1.746, Validation accuracy:53.15%
Epoch 13: iteration 34/35
Epoch 13 --- loss:0.071, Epoch accuracy:95.47%, Validation loss:1.948, Validation accuracy:52.45%
Epoch 14: iteration 34/35
Epoch 14 --- loss:0.028, Epoch accuracy:97.30%, Validation loss:1.736, Validation accuracy:63.64%
Epoch 15: iteration 34/35
Epoch 15 --- loss:0.009, Epoch accuracy:97.56%, Validation loss:1.789, Validation accuracy:60.14%
Epoch 16: iteration 34/35
Epoch 16 --- loss:0.008, Epoch accuracy:97.56%, Validation loss:1.698, Validation accuracy:64.34%
Epoch 17: iteration 34/35
Epoch 17 --- loss:0.006, Epoch accuracy:97.65%, Validation loss:1.770, Validation accuracy:61.54%
Epoch 18: iteration 34/35
Epoch 18 --- loss:0.006, Epoch accuracy:97.56%, Validation loss:1.773, Validation accuracy:61.54%
Epoch 19: iteration 34/35
Epoch 19 --- loss:0.005, Epoch accuracy:97.56%, Validation loss:2.122, Validation accuracy:55.24%
Epoch 20: iteration 34/35
Epoch 20 --- loss:0.007, Epoch accuracy:97.47%, Validation loss:1.989, Validation accuracy:58.04%
Epoch 21: iteration 34/35
Epoch 21 --- loss:0.007, Epoch accuracy:97.38%, Validation loss:1.814, Validation accuracy:64.34%
Epoch 22: iteration 34/35
Epoch 22 --- loss:0.003, Epoch accuracy:97.65%, Validation loss:1.729, Validation accuracy:64.34%
Epoch 23: iteration 34/35
Epoch 23 --- loss:0.002, Epoch accuracy:97.65%, Validation loss:1.762, Validation accuracy:62.94%
Epoch 24: iteration 34/35
Epoch 24 --- loss:0.002, Epoch accuracy:97.65%, Validation loss:1.836, Validation accuracy:63.64%
Epoch 25: iteration 34/35
Epoch 25 --- loss:0.002, Epoch accuracy:97.65%, Validation loss:1.760, Validation accuracy:63.64%
Epoch 26: iteration 34/35
Epoch 26 --- loss:0.002, Epoch accuracy:97.65%, Validation loss:1.822, Validation accuracy:62.24%
Epoch 27: iteration 34/35
Epoch 27 --- loss:0.002, Epoch accuracy:97.65%, Validation loss:1.766, Validation accuracy:60.84%
Epoch 28: iteration 34/35
Epoch 28 --- loss:0.001, Epoch accuracy:97.65%, Validation loss:1.783, Validation accuracy:61.54%
Epoch 29: iteration 34/35
Epoch 29 --- loss:0.002, Epoch accuracy:97.65%, Validation loss:1.800, Validation accuracy:61.54%
Epoch 30: iteration 34/35
Epoch 30 --- loss:0.001, Epoch accuracy:97.65%, Validation loss:1.823, Validation accuracy:61.54%
Epoch 31: iteration 34/35
Epoch 31 --- loss:0.001, Epoch accuracy:97.65%, Validation loss:1.760, Validation accuracy:62.24%
Epoch 32: iteration 34/35
Epoch 32 --- loss:0.001, Epoch accuracy:97.65%, Validation loss:1.803, Validation accuracy:61.54%
Epoch 33: iteration 34/35
Epoch 33 --- loss:0.001, Epoch accuracy:97.65%, Validation loss:1.774, Validation accuracy:61.54%
Epoch 34: iteration 34/35
Epoch 34 --- loss:0.001, Epoch accuracy:97.65%, Validation loss:1.787, Validation accuracy:62.94%
Epoch 35: iteration 34/35
Epoch 35 --- loss:0.002, Epoch accuracy:97.56%, Validation loss:1.675, Validation accuracy:62.24%
Epoch 36: iteration 34/35
Epoch 36 --- loss:0.004, Epoch accuracy:97.56%, Validation loss:1.841, Validation accuracy:61.54%
Epoch 37: iteration 34/35
Epoch 37 --- loss:0.001, Epoch accuracy:97.65%, Validation loss:1.764, Validation accuracy:60.14%
Epoch 38: iteration 34/35
Epoch 38 --- loss:0.001, Epoch accuracy:97.65%, Validation loss:1.738, Validation accuracy:60.84%
Epoch 39: iteration 34/35
Epoch 39 --- loss:0.001, Epoch accuracy:97.65%, Validation loss:1.753, Validation accuracy:62.94%
Epoch 40: iteration 34/35
Epoch 40 --- loss:0.001, Epoch accuracy:97.65%, Validation loss:1.672, Validation accuracy:64.34%
Epoch 41: iteration 34/35
Epoch 41 --- loss:0.002, Epoch accuracy:97.65%, Validation loss:1.842, Validation accuracy:60.84%
Epoch 42: iteration 34/35
Epoch 42 --- loss:0.002, Epoch accuracy:97.56%, Validation loss:1.727, Validation accuracy:62.94%
Epoch 43: iteration 34/35
Epoch 43 --- loss:0.001, Epoch accuracy:97.65%, Validation loss:1.851, Validation accuracy:64.34%
Epoch 44: iteration 34/35
Epoch 44 --- loss:0.001, Epoch accuracy:97.65%, Validation loss:1.855, Validation accuracy:62.24%
Epoch 45: iteration 34/35
Epoch 45 --- loss:0.001, Epoch accuracy:97.65%, Validation loss:1.836, Validation accuracy:62.94%
Epoch 46: iteration 34/35
Epoch 46 --- loss:0.001, Epoch accuracy:97.65%, Validation loss:1.821, Validation accuracy:63.64%
Epoch 47: iteration 34/35
Epoch 47 --- loss:0.004, Epoch accuracy:97.56%, Validation loss:1.889, Validation accuracy:61.54%
Epoch 48: iteration 34/35
Epoch 48 --- loss:0.003, Epoch accuracy:97.47%, Validation loss:1.873, Validation accuracy:60.14%
Epoch 49: iteration 34/35
Epoch 49 --- loss:0.002, Epoch accuracy:97.65%, Validation loss:1.710, Validation accuracy:63.64%
    """