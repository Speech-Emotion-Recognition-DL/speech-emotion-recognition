from torch.utils.data import Dataset
import numpy as np
import torch
import torchaudio

import pandas as pd
import matplotlib.pyplot as plt
# import librosa
# import librosa.display
import warnings
from sklearn.preprocessing import StandardScaler
from collections import Counter
from project import create_csv

from audiomentations import Compose, AddGaussianNoise, TimeMask, PitchShift, BandStopFilter, Shift, Gain, RoomSimulator

augment = Compose([
    Shift(min_fraction=((1 / 3) / 2) / 10, max_fraction=((1 / 3) / 2), rollover=False, fade=False, p=0.5),
    Gain(min_gain_in_db=-10, max_gain_in_db=10, p=0.5),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
])

warnings.filterwarnings('ignore')  # matplot lib complains about librosa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
bundle = torchaudio.pipelines.WAV2VEC2_BASE
# bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

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

    def tensor_4D(self, X_, features, Y):
        # print()
        # print(X.shape)
        X_ = np.stack([np.expand_dims(t.cpu().numpy(), axis=0) for t in features], axis=0)
        # print(X.shape)
        X_ = np.squeeze(X_, axis=1)
        # print(X.shape)
        # convert emotion labels from list back to numpy arrays for PyTorch to work with
        Y = np.array(Y)
        # print(Y)
        return X_, Y

    def feature_Scaling(self, X_train):
        scaler = StandardScaler()

        N, C, H, W = X_train.shape

        X_train = np.reshape(X_train, (N, -1))
        X_train = scaler.fit_transform(X_train)

        # Transform back to NxCxHxW 4D tensor format
        X_train = np.reshape(X_train, (N, C, H, W))
        return X_train

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

    def augment_balanced_data(self, X_train, Y_train):

        print("\n ---------------------- ")
        print(Y_train.size)
        print()
        # Find the total number of instances of each class in the training set
        class_counts = Counter(Y_train)
        print("class_count", class_counts)

        # Initialize a dictionary to store the number of instances of each class to augment
        num_to_augment = {}

        # Iterate through each class
        for cls, count in class_counts.items():
            # Calculate the number of instances of this class to augment
            num_to_augment[cls] = count // 2

        print("num_to_augment ", num_to_augment)

        # Initialize lists to store the augmented instances
        augmented_X, augmented_Y = [], []

        # Iterate through each class
        for cls, count in num_to_augment.items():
            # Find the indices of the instances of this class
            indices = np.where(Y_train == cls)[0]
            # print(indices)

            # Randomly select the indices to augment
            to_augment = np.random.choice(indices, size=count, replace=False)

            # Retrieve the corresponding instances from the training set
            X_to_augment = X_train[to_augment, :]
            # print(X_to_augment)

            # Y_to_augment = Y_train[to_augment]

            # Apply data augmentation to the retrieved instances
            augmented_X_cls = augment(X_to_augment, sample_rate)
            augmented_Y_cls = np.array([cls] * len(augmented_X_cls))

            # Add the augmented instances to the lists
            augmented_X.append(augmented_X_cls)
            augmented_Y.append(augmented_Y_cls)

        # Concatenate the augmented instances with the rest of the training set
        # Concatenate the augmented instances with the rest of the training set
        print("x train shape : ", X_train.shape)
        print(augmented_X)
        augmented_X = np.array(augmented_X)
        print("dddddd", augmented_X)

        augmented_Y = np.array(augmented_Y)

        X_train_new = np.concatenate([X_train, augmented_X], axis=0)
        Y_train_new = np.concatenate([Y_train, augmented_Y], axis=0)

        return X_train_new, Y_train_new

    def get(self):
        waveforms, emotions = self.load_data()

        train_XY, valid_XY, test_XY = self.split_data(waveforms, emotions)

        train_X = train_XY[0]
        valid_X = valid_XY[0]
        test_X = test_XY[0]

        train_Y = train_XY[1]
        valid_Y = valid_XY[1]
        test_Y = test_XY[1]

        train_X, train_Y = self.augment_balanced_data(train_X, train_Y)
        print("train_X.size " , train_X.size)
        features_train_X = self.feature_extraction(train_X)
        features_valid_X = self.feature_extraction(valid_X)
        features_test_X = self.feature_extraction(test_X)

        train_X, train_Y = self.tensor_4D(train_X, features_train_X, train_Y)
        valid_X, valid_Y = self.tensor_4D(valid_X, features_valid_X, valid_Y)
        test_X, test_Y = self.tensor_4D(test_X, features_test_X, test_Y)

        train_X = self.feature_Scaling(train_X)
        valid_X = self.feature_Scaling(valid_X)
        test_X = self.feature_Scaling(test_X)

        return train_X, train_Y, valid_X, valid_Y, test_X, test_Y


if __name__ == '__main__':
    dm = DataManagement()
    dm.get()

    # waveforms, emotions = dm.load_data()
    #
    # print(f'Waveforms set: {len(waveforms)} samples')
    # # we have 1440 waveforms but we need to know their length too; should be 3 sec * 48k = 144k
    # print(f'Waveform signal length: {len(waveforms[0])}')
    # print(f'Emotions set: {len(emotions)} sample labels')
    #
    # train_XY, valid_XY, test_XY = dm.split_data(waveforms, emotions)
    #
    # train_X = train_XY[0]
    # valid_X = valid_XY[0]
    # test_X = test_XY[0]
    # train_Y = train_XY[1]
    # valid_Y = valid_XY[1]
    # test_Y = test_XY[1]
    #
    # features_train_X = dm.feature_extraction(train_X)
    # features_valid_X = dm.feature_extraction(valid_X)
    # features_test_X = dm.feature_extraction(test_X)
    #
    # train_X, train_Y = dm.tensor_4D(train_X, features_train_X, train_Y)
    # valid_X, valid_Y = dm.tensor_4D(valid_X, features_valid_X, valid_Y)
    # test_X, test_Y = dm.tensor_4D(test_X, features_test_X, test_Y)
    #
    # train_X = dm.feature_Scaling(train_X)
    # valid_X = dm.feature_Scaling(valid_X)
    # test_X = dm.feature_Scaling(test_X)
    #
    # print(
    #     f'Shape of 4D feature array for input tensor: {train_X.shape} train, {valid_X.shape} validation, {test_X.shape} test')
