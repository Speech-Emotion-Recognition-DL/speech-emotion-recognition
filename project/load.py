from cnn_model import Convolutional_Neural_Network

import torch
import numpy as np
from myDataset import SoundDataset
import torchaudio


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

def predict(model, test_loader):
    model.eval()
    pred_list = []
    target = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            # Get the model's predictions for the input batch
            y_pred = model(x_batch)
            # Append the predicted class (the one with the highest probability) to the list of predictions
            pred_list.append(y_pred.argmax(dim=1))
            # Append the true class to the list of targets
            target.append(y_batch)
    # Convert the lists of predictions and targets to numpy arrays
    pred_array = np.array(pred_list)
    target_array = np.array(target)
    # Calculate the accuracy as the percentage of correct predictions
    accuracy = (pred_array == target_array).mean() * 100
    return accuracy




if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    # # instantiating our dataset object and create data loader
    # mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    #     sample_rate=SAMPLE_RATE,
    #     n_fft=1024,
    #     hop_length=512,
    #     n_mels=64
    # )
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    model_wav2vec = bundle.get_model().to(device)

    # usd = SoundDataset(ANNOTATIONS_FILE,
    #                    model_wav2vec,
    #                    mel_spectrogram,
    #                    SAMPLE_RATE,
    #                    NUM_SAMPLES,
    #                    device)

    # Load the saved model
    saved_state_dict = torch.load('model.pt')

    # Create a new instance of the model class
    model = Convolutional_Neural_Network()

    # Load the saved state dictionary into the model
    model.load_state_dict(saved_state_dict)


    # Load the saved dataset and dataloader
    test_dataset = torch.load('test_dataset.pt')
    test_dataloader = torch.load('test_dataloader.pt')

    val = validate(model.to(device), test_dataloader, device)
    print(val)

    # acu = predict(model, test_dataloader)
    # print(acu)

