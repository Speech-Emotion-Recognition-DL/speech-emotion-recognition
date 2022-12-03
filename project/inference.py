import torch
import torchaudio

from cnn import CNNNetwork
from myDataset import SoundDataset
from train import ANNOTATIONS_FILE, SAMPLE_RATE, NUM_SAMPLES

int2emotion = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

class_mapping = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised"
]

"""
model.eval() is a kind of switch for some specific layers/parts of the model that behave differently
 during training and inference (evaluating) time. For example, 
 Dropouts Layers, BatchNorm Layers etc. You need to turn off them d
 uring model evaluation, and .eval() will do it for you. In addition, 
 the common practice for evaluating/validation is using torch.no_grad() 
in pair with model.eval() to turn off gradients computation:"""


def predict(model, input, target, class_mapping):
    model.eval()  #
    with torch.no_grad():
        predictions = model(input)
        # Tensor (the number of sample that we are passing to the model = 1, the number of classes that the modle tries to predict = 10)
        # Tensor ( 1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        # We are interested in the index that has the highest value.
        # So that index will correspond to the class that we want to predict.
        predicted_index = predictions[0].argmax(0)
        if predicted_index == 8:
            predicted_index = 7

        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected


if __name__ == "__main__":

    # load back the model
    cnn = CNNNetwork()

    # load the file from the train section
    state_dict = torch.load("feedforwardnet.pth")
    cnn.load_state_dict(state_dict)

    # load the dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model_wev2vec = bundle.get_model().to(device)
    usd = SoundDataset(ANNOTATIONS_FILE,
                       model_wev2vec,
                       mel_spectrogram,
                       SAMPLE_RATE,
                       NUM_SAMPLES,
                       "cpu")

    # get a sample from the dataset for inference
    input = usd[0][0]
    target = usd[0][1]  # [batch size, num_channels, fr, time]
    input.unsqueeze_(0)

    # make an inference
    predicted, expected = predict(cnn, input, target,
                                  class_mapping)
    print(f"Predicted :  '{predicted}', expected: '{expected}'")

    # for i in range(50):
    #     input = usd[i][0]
    #     target = usd[i][1]  # [batch size, num_channels, fr, time]
    #     input.unsqueeze_(0)
    #
    #     # make an inference
    #     predicted, expected = predict(cnn, input, target,
    #                                   class_mapping)
    #     print(f"Predicted {i}:  '{predicted}', expected: '{expected}'")
    #     print("\n -------------------------------------------------- \n")
