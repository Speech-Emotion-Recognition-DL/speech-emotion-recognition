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


def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected


if __name__ == "__main__":
    # load back the model
    cnn = CNNNetwork()
    # cnn = cnn.to('cuda')
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
    print("---------result------------")

    for i in range(52):
        input = usd[i][0]
        target = usd[i][1]  # [batch size, num_channels, fr, time]
        input.unsqueeze_(0)

        # make an inference
        predicted, expected = predict(cnn, input, target,
                                      class_mapping)
        print(f"Predicted {i}:  '{predicted}', expected: '{expected}'")
        print("\n -------------------------------------------------- \n")
