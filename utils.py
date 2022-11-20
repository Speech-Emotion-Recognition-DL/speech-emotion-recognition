# import soundfile  # read the audio file
# import numpy as np
# import librosa  # to extract speech features
# import matplotlib.pyplot as plt
# from Ipython.display import Audio
# from librosa import display
# import glob
# import os
# import pickle  # to save model after training
# from sklearn.model_selection import train_test_split  # for splitting training and testing
# from sklearn.neural_network import MLPClassifier  # multi-layer perceptron model
# from sklearn.metrics import accuracy_score  # to measure how good we are
#
# # all emotions on RAVDESS dataset
# int2emotion = {
#     "01": "neutral",
#     "02": "calm",
#     "03": "happy",
#     "04": "sad",
#     "05": "angry",
#     "06": "fearful",
#     "07": "disgust",
#     "08": "surprised"
# }
#
# # we allow only these emotions ( feel free to tune this on your need )
# AVAILABLE_EMOTIONS = {
#     "angry",
#     "sad",
#     "neutral",
#     "happy"
# }
#
#
# def extract_feature(file_name, **kwargs):
#     """
#         Extract feature from audio file `file_name`
#             Features supported:
#                 - MFCC (mfcc)
#                 - Chroma (chroma)
#                 - MEL Spectrogram Frequency (mel)
#                 - Contrast (contrast)
#                 - Tonnetz (tonnetz)
#             e.g:
#             `features = extract_feature(path, mel=True, mfcc=True)`
#         """
#     mfcc = kwargs.get("mfcc")
#     chroma = kwargs.get("chroma")
#     mel = kwargs.get("mel")
#     contrast = kwargs.get("contrast")
#     tonnetz = kwargs.get("tonnetz")
#
#     with soundfile.SoundFile(file_name) as sound_file:
#         X = sound_file.read(dtype="float32")
#         sample_rate = sound_file.samplerate
#         if chroma or contrast:
#             stft = np.abs(librosa.stft(X))  # Short-time Fourier transform
#         result = np.array([])
#         if mfcc:
#             mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
#             result = np.hstack((result, mfccs))
#         if chroma:
#             chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
#             result = np.hstack((result, chroma))
#         if mel:
#             mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
#             result = np.hstack((result, mel))
#         if contrast:
#             contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
#             result = np.hstack((result, contrast))
#         if tonnetz:
#             tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
#             result = np.hstack((result, tonnetz))
#     return result
#
#
# once = True
#
#
# def load_data(test_size=0.2):
#     global once
#     X, y = [], []
#     for file in glob.glob("../../../PycharmProjects/pyTorch/project/data/Actor_*/*.wav"):
#         # get the base name of the audio file
#         base_name = os.path.basename(file)
#         # get the emotion label
#         emotion = int2emotion[base_name.split("-")[2]]
#         # print(emotion)
#         # we allow only AVAILABLE_EMOTIONS we set
#         if emotion not in AVAILABLE_EMOTIONS:
#             continue
#         # extract speech features
#         features = extract_feature(file, mfcc=True, chroma=True, mel=True)
#         if once:
#             print(features)
#             print(emotion)
#             print(base_name)
#             data, sp = librosa.load(file)
#             plt.figure(figsize=(10, 4))
#             plt.title("test", size=20)
#             librosa.display.waveplot(data, sp)
#             plt.show()
#             once = False
#         # add to data
#         X.append(features)
#         y.append(emotion)
#     # split the data to training and testing and return it
#     return train_test_split(np.array(X), y, test_size=test_size, random_state=7)  # sklearn method
