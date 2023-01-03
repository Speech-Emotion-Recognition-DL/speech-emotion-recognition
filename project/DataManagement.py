from torch.utils.data import Dataset
import numpy as np
import torch
import torchaudio
# import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import warnings

warnings.filterwarnings('ignore')  # matplot lib complains about librosa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
bundle = torchaudio.pipelines.WAV2VEC2_BASE
model = bundle.get_model().to(device)
sample_rate = bundle.sample_rate
ANNOTATIONS_FILE = 'Train_test_.csv'


class DataManagement:

    def __init__(self):
        self.annotations = pd.read_csv(ANNOTATIONS_FILE)

    def __len__(self):
        """
        the length of our dataframe
        """
        return len(self.annotations)

    def load_data(self):
        """ In this method, we load the data that we will be used from the model,
            The data path & labels is depend on our CSV that we built before,
            Iterate on all the data in CSV by index from total length of the CSV,
            take each label and extract each correlate signal(via signal() method ) to fill emotions and waveforms lists
            Return tuple of (Waveforms ,Emotion).
            """
        waveforms = []
        emotions = []
        index = 0
        while index < self.__len__():
            # get the path from CSV
            audio_sample_path = self._get_audio_sample_path(index)

            # extract the correlate label from CSV
            label = self._get_audio_sample_label(index)
            emotions.append(label)
            # get the waveform from the audio file via torch_audio.load
            waveform = self.signal(audio_sample_path)
            waveforms.append(waveform)
            print('\r' + f"Num of waveforms: {index + 1}/{self.__len__()}", end='')
            index += 1

        print()
        return waveforms, emotions

    def split_data(self):

        # lists that will contain data extracted from signals
        X_train, X_valid, X_test = [], [], []
        # lists that will contain labels extracted from CSV
        Y_train, Y_valid, Y_test = [], [], []

        waveforms_arr = np.array(waveforms)

        # add the indexes of the specific *emotion_num to emotion_indices.
        # the output list "emotion_indices" will be a list contains all the indexes in the emotion list
        # where the specific emotion_num appears.

        emotions_indexes = []
        # for index, emotion in enumerate(emotions):
        #     if emotion ==

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

        """ If the sample rate of the signal is not 16,000, resample."""
        if sr != bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)

        """ In RAVDESS each audio sample have quiet period time in the 0.5 first sec
            first we take each waveform with the shifted 0.5 first sec, and then the total 3 second long further.
            Now each sample have same length.
            """
        half_sec = int(0.5 * sr)  # shift 0.5 sec
        wave = np.array(waveform[0].numpy())
        waveform_homo = np.zeros((int(sample_rate * 3, )))
        waveform_homo[:len(wave[half_sec:half_sec + 3 * sample_rate])] = wave[half_sec:half_sec + 3 * sample_rate]

        return waveform_homo

    """ -------------- Methods for CSV -------------- """

    def _get_audio_sample_path(self, index):
        """
         # getting the value of the cell from the csv file
         # row is the index
         # column is 1
        """
        path = self.annotations.iloc[index, 1]
        return path

    def _get_audio_sample_label(self, index):
        """
         # getting the value of the cell from the csv file
         # row is the index
         # column is 1"
        """
        label = self.annotations.iloc[index, 3]
        return label

    """ --------------------  Methods for Signal Preprocess -------------------- """

    def _mix_down_if_necessary(self, signal):
        """
         Normalization of the audio if it have more than one channels.
        """
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal


if __name__ == '__main__':
    dm = DataManagement()
    waveforms, emotions = dm.load_data()
    print(emotions)
    print(f'Waveforms set: {len(waveforms)} samples')
    # we have 1440 waveforms but we need to know their length too; should be 3 sec * 48k = 144k
    print(f'Waveform signal length: {len(waveforms[0])}')
    print(f'Emotions set: {len(emotions)} sample labels')
