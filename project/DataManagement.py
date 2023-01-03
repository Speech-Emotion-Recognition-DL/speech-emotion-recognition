from torch.utils.data import Dataset
import numpy as np
import torch
import torchaudio
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import warnings; warnings.filterwarnings('ignore') #matplot lib complains about librosa



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

        # todo channel check
        waveforms.append(self.signal(audio_sample_path))

        return waveforms


    def split(self):


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


        #print(waveform.shape)
        # data = waveform.numpy()
        # plt.figure(figsize=(15, 4))
        # plt.subplot(1, 2, 1)
        # librosa.display.waveshow(data, sr=sr)
        #
        # plt.title('before resample')
        # plt.show()


        if sr != bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)
        # data = waveform.numpy()
        # plt.figure(figsize=(15, 4))
        # plt.subplot(1, 2, 1)
        # librosa.display.waveshow(data, sr=sr)
        # plt.title('after resample')
        # plt.show()



        #todo what this part do ?
        waveform_len = len(waveform)
        half_sec = int(0.5 * sr)
        wave = np.array(waveform[0].numpy())
        waveform_homo = np.zeros((int(sample_rate * 3, )))
        waveform_homo[:len(wave[half_sec:half_sec + 3 * sample_rate])] = wave[half_sec:half_sec + 3 * sample_rate]


        # return a single file's waveform
        return waveform_homo


    def _get_audio_sample_path(self, index):
        """
         # getting the value of the cell from the csv file
         # row is the index
         # column is 1
        """
        path = self.annotations.iloc[index, 1]
        return path


if __name__ == '__main__':
    dm = DataManagement()
    print(dm.__getitem__(0))



## 3 emo
"""
Epoch 0: iteration 34/35
Epoch 0 --- loss:0.947, Epoch accuracy:53.26%, Validation loss:0.834, Validation accuracy:59.03%
Epoch 1: iteration 34/35
Epoch 1 --- loss:0.666, Epoch accuracy:66.64%, Validation loss:0.731, Validation accuracy:65.97%
Epoch 2: iteration 34/35
Epoch 2 --- loss:0.529, Epoch accuracy:75.93%, Validation loss:0.634, Validation accuracy:69.44%
Epoch 3: iteration 34/35
Epoch 3 --- loss:0.320, Epoch accuracy:83.93%, Validation loss:0.644, Validation accuracy:73.61%
Epoch 4: iteration 34/35
Epoch 4 --- loss:0.240, Epoch accuracy:87.92%, Validation loss:1.059, Validation accuracy:65.97%
Epoch 5: iteration 34/35
Epoch 5 --- loss:0.153, Epoch accuracy:91.83%, Validation loss:0.870, Validation accuracy:72.92%
Epoch 6: iteration 34/35
Epoch 6 --- loss:0.107, Epoch accuracy:93.74%, Validation loss:0.857, Validation accuracy:69.44%
Epoch 7: iteration 34/35
Epoch 7 --- loss:0.085, Epoch accuracy:94.09%, Validation loss:1.397, Validation accuracy:72.22%
Epoch 8: iteration 34/35
Epoch 8 --- loss:0.062, Epoch accuracy:95.66%, Validation loss:0.751, Validation accuracy:75.69%
Epoch 9: iteration 34/35
Epoch 9 --- loss:0.059, Epoch accuracy:94.61%, Validation loss:1.403, Validation accuracy:70.14%
Epoch 10: iteration 34/35
Epoch 10 --- loss:0.100, Epoch accuracy:93.83%, Validation loss:1.134, Validation accuracy:72.22%
Epoch 11: iteration 34/35
Epoch 11 --- loss:0.099, Epoch accuracy:93.92%, Validation loss:0.725, Validation accuracy:77.08%
Epoch 12: iteration 34/35
Epoch 12 --- loss:0.049, Epoch accuracy:95.92%, Validation loss:0.986, Validation accuracy:76.39%
Epoch 13: iteration 34/35
Epoch 13 --- loss:0.022, Epoch accuracy:96.70%, Validation loss:0.899, Validation accuracy:79.17%
Epoch 14: iteration 34/35
Epoch 14 --- loss:0.027, Epoch accuracy:96.26%, Validation loss:1.291, Validation accuracy:73.61%
Epoch 15: iteration 34/35
Epoch 15 --- loss:0.014, Epoch accuracy:96.96%, Validation loss:1.155, Validation accuracy:73.61%
Epoch 16: iteration 34/35
Epoch 16 --- loss:0.013, Epoch accuracy:97.05%, Validation loss:0.966, Validation accuracy:75.69%
Epoch 17: iteration 34/35
Epoch 17 --- loss:0.022, Epoch accuracy:96.52%, Validation loss:1.681, Validation accuracy:73.61%
Epoch 18: iteration 34/35
Epoch 18 --- loss:0.008, Epoch accuracy:97.13%, Validation loss:1.452, Validation accuracy:76.39%
Epoch 19: iteration 34/35
Epoch 19 --- loss:0.005, Epoch accuracy:97.05%, Validation loss:1.064, Validation accuracy:73.61%
Epoch 20: iteration 34/35
Epoch 20 --- loss:0.006, Epoch accuracy:97.05%, Validation loss:1.236, Validation accuracy:77.78%
Epoch 21: iteration 34/35
Epoch 21 --- loss:0.006, Epoch accuracy:97.22%, Validation loss:1.071, Validation accuracy:75.69%
Epoch 22: iteration 34/35
Epoch 22 --- loss:0.014, Epoch accuracy:96.79%, Validation loss:1.168, Validation accuracy:79.17%
Epoch 23: iteration 34/35
Epoch 23 --- loss:0.004, Epoch accuracy:97.31%, Validation loss:1.173, Validation accuracy:77.78%
Epoch 24: iteration 34/35
Epoch 24 --- loss:0.006, Epoch accuracy:97.13%, Validation loss:0.965, Validation accuracy:78.47%
Epoch 25: iteration 34/35
Epoch 25 --- loss:0.004, Epoch accuracy:97.31%, Validation loss:1.070, Validation accuracy:78.47%
Epoch 26: iteration 34/35
Epoch 26 --- loss:0.002, Epoch accuracy:97.31%, Validation loss:1.065, Validation accuracy:79.17%
Epoch 27: iteration 34/35
Epoch 27 --- loss:0.001, Epoch accuracy:97.31%, Validation loss:1.095, Validation accuracy:79.17%
Epoch 28: iteration 34/35
Epoch 28 --- loss:0.000, Epoch accuracy:97.31%, Validation loss:1.147, Validation accuracy:79.17%
Epoch 29: iteration 34/35
Epoch 29 --- loss:0.000, Epoch accuracy:97.31%, Validation loss:1.126, Validation accuracy:79.17%
Epoch 30: iteration 34/35
Epoch 30 --- loss:0.000, Epoch accuracy:97.31%, Validation loss:1.104, Validation accuracy:79.17%
Epoch 31: iteration 34/35
Epoch 31 --- loss:0.000, Epoch accuracy:97.31%, Validation loss:1.150, Validation accuracy:79.17%
Epoch 32: iteration 6/35
"""
