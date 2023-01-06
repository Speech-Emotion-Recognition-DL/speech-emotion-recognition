import wave
import pyaudio
import torchaudio
import matplotlib.pyplot as plt
import librosa.display
import torch
from cnn_model_definition import Convolutional_Speaker_Identification
import numpy as np
from DataManagement import DataManagement

# Set the recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 3
WAVE_OUTPUT_FILENAME = "recording.wav"

# # Initialize the PyAudio object
p = pyaudio.PyAudio()

# Open a streaming object to record audio
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Start recording
print("Recording...")
frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

# Stop recording
stream.stop_stream()
stream.close()
p.terminate()

print("Finished recording.")

# Save the recorded audio to a wave file
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

signal, sr = torchaudio.load(
    'recording.wav')
# plot(signal, sr, "original")
data = signal.numpy()

plt.figure()
librosa.display.waveshow(data, sr=sr)
plt.show()

# Load the model's weights and parameters from the file
state_dict = torch.load('model_with_aug64.pt')

# Create a new model object
model = Convolutional_Speaker_Identification()

# Load the weights and parameters into the model
model.load_state_dict(state_dict)

sample_rate = 16000
""" If the sample rate of the signal is not 16,000, resample."""
if sr != sample_rate:
    waveform = torchaudio.functional.resample(signal, sr, sample_rate)

half_sec = int(0.5 * sr)  # shift 0.5 sec
signal = np.array(signal[0].cpu().numpy())
waveform_homo = np.zeros((int(sample_rate * 3, )))
waveform_homo[:len(signal[half_sec:half_sec + 3 * sample_rate])] = signal[half_sec:half_sec + 3 * sample_rate]

waveform_homo = waveform_homo.reshape(1, -1)


# print(waveform.shape)

def feature_mfcc(
        waveform,
        sample_rate,
        n_mfcc=40,
        fft=1024,
        winlen=512,
        window='hamming',
        # hop=256, # increases # of time steps; was not helpful
        mels=128
):
    # Compute the MFCCs for all STFT frames
    # 40 mel filterbanks (n_mfcc) = 40 coefficients
    mfc_coefficients = librosa.feature.mfcc(
        y=waveform,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=fft,
        win_length=winlen,
        window=window,
        # hop_length=hop,
        n_mels=mels,
        fmax=sample_rate / 2
    )
    return mfc_coefficients


##mfcc
mfccs = feature_mfcc(waveform_homo, sample_rate)

mfccs = torch.tensor(mfccs).unsqueeze(0).float()

# Get the prediction from the model
with torch.no_grad():
    output = model(mfccs)
output_tensor, other_variable = output
print(output)
prediction = output_tensor.argmax(dim=1)

# Map the prediction to the corresponding emotion label
emotion_labels = ['neutral', 'positive', 'negative']
emotion = emotion_labels[prediction]
print(f'Predicted emotion: {emotion}')


import os

file_path = 'recording.wav'

if os.path.exists(file_path):
    os.remove(file_path)

