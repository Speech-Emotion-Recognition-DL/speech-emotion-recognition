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

        # # stats of this data
        # self.print_stats(signal, sample_rate=sample_rate)
        # self.plot_waveform(signal, sample_rate)
        # self.plot_specgram(signal, sample_rate)

        duration = signal.shape[1] / sample_rate
        # print(f"The signal has a duration of {duration:.2f} seconds.")


        if duration < 2:
            signal = signal.repeat(1, 2)

        # Trim the signal to the first 2 seconds
        start_time = 0
        end_time = 2
        start_index = int(start_time * sample_rate)
        end_index = int(end_time * sample_rate)
        signal = signal.narrow(1, start_index, end_index - start_index)
        # Save the trimmed signal to a new audio file
        # torchaudio.save("trimmed_audio.wav", signal, sample_rate)

        duration = signal.shape[1] / sample_rate
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

    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    print("Sample Rate:", bundle.sample_rate)

    ANNOTATIONS_FILE = '../project/Train_test_.csv'

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
    signal, label = usd[1]

    a = 1

