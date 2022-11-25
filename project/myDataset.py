import os

import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio

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


class SoundDataset(Dataset):

    def __init__(self,
                 annotations_file,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device):
        self.annotations = pd.read_csv(annotations_file)
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        """
        the length of our dataframe
        """
        return len(self.annotations)

    def __getitem__(self, index):
        """
        Getting loading the waveform of the sample associated
        to a certain index and then at the same time return also the label associated with it.

        for our sound classifier waveform is not enfue
        rather mel spectrogram

        torchaudio.load :
        signal - the waveform
        sample rate - time series of the audio file
        """
        # Todo change the transformer to wev2vec
        # Todo what is the best number of sample rate and num of sample

        # Loads an audio file at a particular index within the dataset
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)

        # transforming the signal(waveform) into mel
        # spectrogram (or pass it into our transportation)

        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)

        # adjusting the audio length
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)

        signal = self.transformation(signal)
        return signal, label

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            # [take the all dimension and leave it completely:,:(1 ,50000)-> (1,22050)]
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        """
        Resample all the data that we load so all
        the data has the same sample rate,so we can ensure at this point
        that the output of what we have in terms of mal spectrogram is
        coherent in its shape in its dimensions
        """
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        """
         Normalization of the audio if it have more than one channels.
        """
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        # getting the value of the cell from the csv file
        # row is the index
        # column is 1
        path = self.annotations.iloc[index, 1]
        return path

    def _get_audio_sample_label(self, index):
        # getting the value of the cell from the csv file
        # row is the index
        # column is 1
        label = self.annotations.iloc[index, 3]
        return label


if __name__ == "__main__":
    ANNOTATIONS_FILE = 'C:/Users/97252/Documents/GitHub/speech-emotion-recognition/project/Train_tess_ravdess.csv'
    SAMPLE_RATE = 16000
    NUM_SAMPLES = 22050

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    # transformer
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,  # frame size
        hop_length=512,
        n_mels=64  # number of mel
    )

    usd = SoundDataset(ANNOTATIONS_FILE,
                       mel_spectrogram,
                       SAMPLE_RATE,
                       NUM_SAMPLES,
                       device)

    print(f"There are {len(usd)} samples in the dataset.")
    signal, label = usd[0]

    a = 1

    # p = pd.read_csv("C:/Users/97252/PycharmProjects/pyTorch/project/Test_tess_ravdess.csv")
    # print(p)