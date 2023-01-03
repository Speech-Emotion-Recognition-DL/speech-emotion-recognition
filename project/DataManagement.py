from torch.utils.data import Dataset
import numpy as np
import torch
import torchaudio
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import warnings;

warnings.filterwarnings('ignore')  # matplot lib complains about librosa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
bundle = torchaudio.pipelines.WAV2VEC2_BASE
model = bundle.get_model().to(device)
sample_rate = bundle.sample_rate
ANNOTATIONS_FILE = 'Train_test_.csv'


class DataManagement(Dataset):

    def __init__(self):
        self.annotations = pd.read_csv(ANNOTATIONS_FILE)

    def __len__(self):
        """
        the length of our dataframe
        """
        return len(self.annotations)

    def __getitem__(self, index):
        waveforms = []

        audio_sample_path = self._get_audio_sample_path(index)

        waveforms.append(self.signal(audio_sample_path))

        print(waveforms)
        return waveforms


    def split(self):
        pass


    def signal(self, file):
         """
         waveform: a tensor containing the
          audio data of the audio file.
          The shape of this tensor will depend on the
          number of channels and the length of the audio file.

         sr: the sample rate of the audio file, in Hz.
         The sample rate  is the number of samples per second in the audio file.
         """
         waveform, sr = torchaudio.load(file)

         waveform = self._mix_down_if_necessary(waveform)

         # print(waveform.shape)
         # data = waveform.numpy()
         # plt.figure(figsize=(15, 4))
         # plt.subplot(1, 2, 1)
         # librosa.display.waveshow(data, sr=sr)
         #
         # plt.title('before resample')
         # plt.show()
         # print(waveform)
         if sr != bundle.sample_rate:
             waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)
         data = waveform.numpy()
         # plt.figure(figsize=(15, 4))
         # plt.subplot(1, 2, 1)
         # librosa.display.waveshow(data, sr=sr)
         # plt.title('before resample')
         # plt.show()

         # print(waveform)
         # todo what this part do ?
         waveform_len = len(waveform)
         half_sec = int(0.5 * sr)  # shift 0.5 sec
         wave = np.array(waveform[0].numpy())
         waveform_homo = np.zeros((int(sample_rate * 3, )))
         waveform_homo[:len(wave[half_sec:half_sec + 3 * sample_rate])] = wave[half_sec:half_sec + 3 * sample_rate]

         # plt.figure(figsize=(15, 4))
         # plt.subplot(1, 2, 1)
         # librosa.display.waveshow(waveform_homo, sr=sr)
         # plt.title('after resample')
         # plt.show()

         # return a single file's waveform
         # print(waveform_homo.shape)
         # print(waveform[0])
         return waveform_homo


    def _get_audio_sample_path(self, index):
        """
         # getting the value of the cell from the csv file
         # row is the index
         # column is 1
        """
        path = self.annotations.iloc[index, 1]
        return path


    def _mix_down_if_necessary(self, signal):
        """
         Normalization of the audio if it have more than one channels.
        """
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal


if __name__ == '__main__':
    dm = DataManagement()
    i = 0
    while i < dm.__len__():
        dm.__getitem__(i)
        i += 1
