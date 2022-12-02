import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

from myDataset import SoundDataset
from cnn import CNNNetwork
from project.cnn_model_definition import Convolutional_Speaker_Identification

"""
Batch size is a term used in machine learning and refers to the number of training examples utilized in one iteration"""
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.001

ANNOTATIONS_FILE = 'C:/Users/97252/Documents/GitHub/speech-emotion-recognition/project/Train_tess_ravdess.csv'
SAMPLE_RATE = 16000
NUM_SAMPLES = 22050


def create_data_loader(train_data, batch_size):
    """
    Data loader is a class that we can use to wrap a data set in our case te train data
    and it will allow us to fetch data to load data in batches

    so it allow us to load data sets that are heavy on the memory without having any issues

    """
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):

    """
    Loop through all the samples in the dataset and at each iteration we'll get a new batch of samples
    and this where the data loader comes in very handy because its an issue ball which will return us both the input and
    the target or in other word the x's and y's for one batch at each iteration`

    """
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
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
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model_wev2vec = bundle.get_model().to(device)
    usd = SoundDataset(ANNOTATIONS_FILE,
                       model_wev2vec,
                       mel_spectrogram,
                       SAMPLE_RATE,
                       NUM_SAMPLES,
                       device)

    print(usd.__getitem__(0))
    # train_dataloader = create_data_loader(usd, BATCH_SIZE)
    train_dataloader = create_data_loader(usd, 16)

    # construct model and assign it to device
    cnn = CNNNetwork().to(device)
    # cnn = Convolutional_Speaker_Identification().to(device)

    #print(cnn)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(cnn.cuda(), train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(cnn.state_dict(), "feedforwardnet.pth")
    print("Trained feed forward net saved at feedforwardnet.pth")
