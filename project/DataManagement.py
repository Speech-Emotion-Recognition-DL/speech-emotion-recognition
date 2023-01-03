from torch.utils.data import Dataset
import numpy as np
import torch
import torchaudio
# import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
# import librosa
# import librosa.display
import warnings

from project import create_csv

warnings.filterwarnings('ignore')  # matplot lib complains about librosa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# bundle = torchaudio.pipelines.WAV2VEC2_BASE
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

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

    """ -------------- Methods for Data Manipulations -------------- """

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

    def split_data(self, waveforms, emotions):
        """  """

        # lists that will contain data extracted from signals
        X_train, X_valid, X_test = [], [], []
        # lists that will contain labels extracted from CSV
        Y_train, Y_valid, Y_test = [], [], []

        waveforms_arr = np.array(waveforms)

        for emotion_number in range(len(create_csv.emotions_dict_3)):

            # add the indexes of the specific *emotion_num to emotion_indices.
            # the output list "emotion_indices" will be a list contains all the indexes in the emotion list
            # where the specific emotion_num appears.

            emotions_indexes = []
            for index, emotion in enumerate(emotions):
                if emotion == emotion_number:
                    emotions_indexes.append(index)

            # shuffle emotion indexes list
            np.random.seed(69)
            emotions_indexes = np.random.permutation(emotions_indexes)

            # store the length of emotion indexes list
            emotion_t_len = len(emotions_indexes)

            # split and store all the emotion indexes for train/val/test as 80/10/10.
            # train is 80% of data
            train_indexes = emotions_indexes[:int(0.8 * emotion_t_len)]
            # validation is 80% - 90% of the data
            val_indexes = emotions_indexes[int(0.8 * emotion_t_len): int(0.9 * emotion_t_len)]
            # test is 90% to 100% of the data
            test_indexes = emotions_indexes[int(0.9 * emotion_t_len):]

            """ For each X,Y train/val/test,
                we add the correlated data from waveforms depend of the indexes we extracted before
                 """
            # print(train_indexes, )
            X_train.append(waveforms_arr[train_indexes, :])
            Y_train.append(np.array([emotion_number] * len(train_indexes), dtype=np.int32))

            X_valid.append(waveforms_arr[val_indexes, :])
            Y_valid.append(np.array([emotion_number] * len(val_indexes), dtype=np.int32))

            X_test.append(waveforms_arr[test_indexes, :])
            Y_test.append(np.array([emotion_number] * len(test_indexes), dtype=np.int32))

        """ concatenate all the waveforms we added each time to the X train/valid/test to one array"""
        X_train = np.concatenate(X_train, axis=0)
        X_valid = np.concatenate(X_valid, axis=0)
        X_test = np.concatenate(X_test, axis=0)

        """ concatenate all the emotions we added each time to the Y train/valid/test to one array """
        Y_train = np.concatenate(Y_train, axis=0)
        Y_valid = np.concatenate(Y_valid, axis=0)
        Y_test = np.concatenate(Y_test, axis=0)

        # check shape of each set
        print(f'Training waveforms:{X_train.shape}, y_train:{Y_train.shape}')
        print(f'Validation waveforms:{X_valid.shape}, y_valid:{Y_valid.shape}')
        print(f'Test waveforms:{X_test.shape}, y_test:{Y_test.shape}')

        # Return 3 tuples, Each tuple represent (X_, Y_) for train/valid/test.
        return (X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test)

    """ -------------- Methods for Features Extraction -------------- """

    def feature_extraction(self, x_waveforms):
        features = []
        file_count = 0
        for waveform in x_waveforms:
            waveform = waveform.reshape(1, -1)
            # print(waveform.shape)

            # convert the waveform to dtype:float Tensor
            wave_tensor = torch.from_numpy(waveform).float()
            wave_tensor = wave_tensor.to(device)

            with torch.inference_mode():
                feat, _ = model.extract_features(wave_tensor)
            with torch.inference_mode():
                emission, _ = model(wave_tensor)
                features.append(emission)
                print('\r' + f' Processed {file_count}/{len(x_waveforms)} waveforms', end='')
                file_count += 1

        return features

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
        waveform = waveform.to(device)

        """ If the sample rate of the signal is not 16,000, resample."""
        if sr != bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)

        """ In RAVDESS each audio sample have quiet period time in the 0.5 first sec
            first we take each waveform with the shifted 0.5 first sec, and then the total 3 second long further.
            Now each sample have same length.
            """
        half_sec = int(0.5 * sr)  # shift 0.5 sec
        wave = np.array(waveform[0].cpu().numpy())
        waveform_homo = np.zeros((int(sample_rate * 3, )))
        waveform_homo[:len(wave[half_sec:half_sec + 3 * sample_rate])] = wave[half_sec:half_sec + 3 * sample_rate]

        return waveform_homo

    def plot_emission(self, emission):
        # plot the classification results
        plt.imshow(emission[0].cpu().T)
        plt.title("Classification result")
        plt.ylabel("Frame (time-axis)")
        plt.xlabel("Class")
        plt.show()

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

    print(f'Waveforms set: {len(waveforms)} samples')
    # we have 1440 waveforms but we need to know their length too; should be 3 sec * 48k = 144k
    print(f'Waveform signal length: {len(waveforms[0])}')
    print(f'Emotions set: {len(emotions)} sample labels')

    train_XY, valid_XY, test_XY = dm.split_data(waveforms, emotions)

    train_X = train_XY[0]
    valid_X = valid_XY[0]
    test_X = test_XY[0]

    features = dm.feature_extraction(train_X)
    print()
    print(features[0].shape)
    dm.plot_emission(features[0])
