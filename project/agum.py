import random

import numpy as np
import torchaudio
import soundfile as sf
import torch
import matplotlib.pyplot as plt
import librosa.display
from audiomentations import Compose, AddGaussianNoise, TimeMask, PitchShift, BandStopFilter, Shift, Gain, RoomSimulator




# adding white noise
def add_white_noise(signal, noise_factor):
    """
    signalL:numpy array the waveform
    noise_factor : the multiplier that is going to decide
    how much noise we want in our signal or mixes in with our signal

    1. create some noise
    2. add to the augmented signal the noise

    """
    # vector creating using a gaussian distribution
    # center_around = 0,
    # standart_deviation = signal.std(),
    # length = signal.size

    signal = signal.numpy()
    noise = np.random.normal(0, signal.std(), signal.size)
    augmented_signal = signal + noise * noise_factor
    return augmented_signal


def plot(signal, sr, title):
    data = signal.numpy()
    # fig, ax = plt.subplots(nrows=2)
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 2, 1)
    librosa.display.waveshow(data, sr=sr)
    plt.title(title)

    plt.show()
    # print(waveform)


#
# # time stretch
# def time_stretch(signal, stretch_rate):
#     """
#     stretch_rate: ho much we want to slow down or speed up the our signal
#     If rate > 1, then the signal is sped up. If rate < 1, then the signal is slowed down.
#     """
#     signal = signal.numpy()
#     return librosa.effects.time_stretch(signal, stretch_rate)
#
#
# # pith scaling
# def pith_scaling(signal, sr, num_semitones):
#     """
#     num_semitones:scale up or down signal
#     positive -> up
#     negative -> down
#
#     example: num_semitones = 2 -> from c major the scale is going up to d major
#     """
#     signal = signal.numpy()
#     return librosa.effects.pitch_shift(signal, sr, num_semitones)
#
#
# # polarity inversion
# def invert_polarity(signal):
#     """
#     change the signal direction(up and down)
#     """
#     signal = signal.numpy()
#     return signal * -1
#
#
# # random gain
# def random_gain(signal, min_gain_factor, max_gain_factor):
#     gain_factor = random.uniform(min_gain_factor, max_gain_factor)
#     signal = signal.numpy()
#     return signal * gain_factor


augment1 = Compose([
    Shift(min_fraction=((1 / 3) / 2) / 10, max_fraction=((1 / 3) / 2), rollover=False, fade=False, p=0.5),
    Gain(min_gain_in_db=-10, max_gain_in_db=10, p=0.5),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
])

if __name__ == '__main__':
    signal, sr = torchaudio.load(
        'C:/Users/97252/Documents/GitHub/speech-emotion-recognition/project/data/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav')
    # plot(signal, sr, "original")
    data = signal.numpy()

    fig, ax = plt.subplots(nrows=2)
    librosa.display.waveshow(data, sr=sr, ax=ax[0])
    ax[0].set(title="original")

    # augmented_signal = random_gain(signal,2,4)
    augmented_signal = augment1(signal.numpy(), sr)

    librosa.display.waveshow(augmented_signal, sr=sr, ax=ax[1])
    ax[1].set(title="augmented_signal")
    plt.show()
    augmented_signal = torch.tensor(augmented_signal)
    torchaudio.save('stretched_signal.wav', augmented_signal, sr)
