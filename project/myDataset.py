import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
# import librosa
import matplotlib
import matplotlib.pyplot as plt
# import torchaudio.transforms as transforms
# import torchvision.transforms as transforms
import torchvision.transforms as transforms
import random



from audiomentations import Compose, AddGaussianNoise, TimeMask, PitchShift, BandStopFilter
import numpy as np
# from datasets import load_metric
from audiomentations.augmentations.mp3_compression import Mp3Compression
import soundfile as sf #save data as wav file
import os
import glob
from pathlib import Path
# import audio2numpy as a2n


# Change p = 0 for augmentations you dont want to use and p = 1 to augmentation you want
augment1 = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.03, p=1),
    TimeMask(min_band_part=0.0, max_band_part=0.15, p=0), # masks too much time in many cases
    PitchShift(min_semitones=-6, max_semitones=8, p=0),
    BandStopFilter(min_center_freq = 60, max_center_freq = 2500, min_bandwidth_fraction = 0.1, max_bandwidth_fraction = 0.4, p=0)
])

augment2 = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.03, p=0),
    TimeMask(min_band_part=0.0, max_band_part=0.15, p=1), # masks too much time in many cases
    PitchShift(min_semitones=-6, max_semitones=8, p=0),
    BandStopFilter(min_center_freq = 60, max_center_freq = 2500, min_bandwidth_fraction = 0.1, max_bandwidth_fraction = 0.4, p=0)
])

augment3 = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.03, p=0),
    TimeMask(min_band_part=0.0, max_band_part=0.15, p=0), # masks too much time in many cases
    PitchShift(min_semitones=-6, max_semitones=8, p=1),
    BandStopFilter(min_center_freq = 60, max_center_freq = 2500, min_bandwidth_fraction = 0.1, max_bandwidth_fraction = 0.4, p=0)
])

augment4 = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.03, p=0),
    TimeMask(min_band_part=0.0, max_band_part=0.15, p=0), # masks too much time in many cases
    PitchShift(min_semitones=-6, max_semitones=8, p=0),
    BandStopFilter(min_center_freq = 60, max_center_freq = 2500, min_bandwidth_fraction = 0.1, max_bandwidth_fraction = 0.4, p=1)
])


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
                 model,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device,
                 augmentations):
        self.annotations = pd.read_csv(annotations_file)
        self.model = model
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.augmentations = augmentations


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

        # Loads an audio file at a particular index within the dataset
        audio_sample_path = self._get_audio_sample_path(index)
        # print(audio_sample_path)
        label = self._get_audio_sample_label(index)

        signal, sample_rate = torchaudio.load(audio_sample_path)
        #print(sample_rate)

        # Generate a random integer between 0 and 4 (inclusive)
        num = random.randint(0, 2)
       # print(num)
       #  if self.augmentations:
       #      signal = augment1(signal, self.num_samples)
            # if num == 1:
            #     signal = augment1(signal, self.num_samples)
            # elif num == 2:
            #     signal = augment2(signal, self.num_samples)
            # elif num == 3:
            #     signal = augment3(signal, self.num_samples)
            # elif num == 4:
            #     signal = augment3(signal, self.num_samples)


        # if self.augmentations == False:
        #     print("false")
            #signal = augment(signal,self.num_samples)
        # elif not self.augmentations:
        #     print("FALSE")

        # signal = augment1(signal,SAMPLE_RATE)

        # # stats of this data
        # self.print_stats(signal, sample_rate=sample_rate)
        # self.plot_waveform(signal, sample_rate)
        # self.plot_specgram(signal, sample_rate)

        # duration = signal.shape[1] / sample_rate
        # # print(f"The signal has a duration of {duration:.2f} seconds.")
        #
        #
        # if duration < 2:
        #     signal = signal.repeat(1, 2)
        #
        # # Trim the signal to the first 2 seconds
        # start_time = 0
        # end_time = 2
        # start_index = int(start_time * sample_rate)
        # end_index = int(end_time * sample_rate)
        # signal = signal.narrow(1, start_index, end_index - start_index)
        # # Save the trimmed signal to a new audio file
        # # torchaudio.save("trimmed_audio.wav", signal, sample_rate)
        #
        # duration = signal.shape[1] / sample_rate
        # print(f"The signal has a duration of {duration:.2f} seconds.")

        # transforming the signal(waveform) into mel
        # spectrogram (or pass it into our transportation)

        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sample_rate)
        signal = self._mix_down_if_necessary(signal)





        # adjusting the audio length
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)

        # print(signal.shape[1])

        features = self.get_features(signal)

        # fig, ax = plt.subplots(len(features), 1, figsize=(16, 4.3 * len(features)))
        # for i, feats in enumerate(features):
        #     ax[i].imshow(feats[0].cpu())
        #     ax[i].set_title(f"Feature from transformer layer {i + 1}")
        #     ax[i].set_xlabel("Feature dimension")
        #     ax[i].set_ylabel("Frame (time-axis)")
        # plt.tight_layout()
        # plt.show()

        with torch.inference_mode():
            emission, _ = self.model(signal)


        # print(emission.shape)
        # plt.imshow(emission[0].cpu().T)
        # plt.title("Classification result")
        # plt.xlabel("Frame (time-axis)")
        # plt.ylabel("Class")
        # plt.show()
        # print("Class labels:", bundle.get_labels())

        #signal = self.transformation(signal)
        # return signal, label
        return emission, label

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
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).cuda()
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

    def get_features(self, signal):
        with torch.inference_mode():
            features, _ = self.model.extract_features(signal)
            return features

    def print_stats(self, signal, sample_rate=None, src=None):
        if src:
            print("-" * 10)
            print("Source:", src)
            print("-" * 10)
        if sample_rate:
            print("Sample Rate:", sample_rate)
        print("Shape:", tuple(signal.shape))
        print("Dtype:", signal.dtype)
        print(f" - Max:     {signal.max().item():6.3f}")
        print(f" - Min:     {signal.min().item():6.3f}")
        print(f" - Mean:    {signal.mean().item():6.3f}")
        print(f" - Std Dev: {signal.std().item():6.3f}")
        print()
        print(signal)
        print()

    def plot_waveform(self, signal, sample_rate, title="Waveform", xlim=None, ylim=None):
        signal = signal.numpy()

        num_channels, num_frames = signal.shape
        time_axis = torch.arange(0, num_frames) / sample_rate

        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].plot(time_axis, signal[c], linewidth=1)
            axes[c].grid(True)
            if num_channels > 1:
                axes[c].set_ylabel(f'Channel {c + 1}')
            if xlim:
                axes[c].set_xlim(xlim)
            if ylim:
                axes[c].set_ylim(ylim)
        figure.suptitle(title)
        plt.show(block=False)

    def plot_specgram(self, signal, sample_rate, title="Spectrogram", xlim=None):
        signal = signal.numpy()

        num_channels, num_frames = signal.shape
        time_axis = torch.arange(0, num_frames) / sample_rate

        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].specgram(signal[c], Fs=sample_rate)
            if num_channels > 1:
                axes[c].set_ylabel(f'Channel {c + 1}')
            if xlim:
                axes[c].set_xlim(xlim)
        figure.suptitle(title)
        plt.show(block=False)


if __name__ == "__main__":

    # bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    print("Sample Rate:", bundle.sample_rate)

    ANNOTATIONS_FILE = 'Train_test_.csv'

    SAMPLE_RATE = bundle.sample_rate
    NUM_SAMPLES = bundle.sample_rate



    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    model = bundle.get_model().to(device)
    # print(model.__class__)

    # transformer
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,  # frame size
        hop_length=512,
        n_mels=64  # number of mel
    )

    # Change p = 0 for augmentations you dont want to use and p = 1 to augmentation you want
    augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.03, p=1),
        TimeMask(min_band_part=0.0, max_band_part=0.15, p=0),  # masks too much time in many cases
        PitchShift(min_semitones=-6, max_semitones=8, p=0),
        BandStopFilter(min_center_freq=60, max_center_freq=2500, min_bandwidth_fraction=0.1, max_bandwidth_fraction=0.4,
                       p=0)
    ])


    usd = SoundDataset(ANNOTATIONS_FILE,
                       model,
                       mel_spectrogram,
                       SAMPLE_RATE,
                       NUM_SAMPLES,
                       device,True)

    print(f"There are {len(usd)} samples in the dataset.")
    signal, label = usd[0]

    a = 1


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
