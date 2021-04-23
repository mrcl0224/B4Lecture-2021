import numpy as np
import scipy as sp
import librosa
import seaborn

fname = "lol_1.wav"
y, sr = librosa.load(fname, sr = 44100)

import matplotlib.pyplot as plt
import librosa.display

#plt.figure()
#plt.figure(figsize=(15, 5))
#librosa.display.waveplot(y, sr)
#plt.show()

window = 1024
sift = 512
i = 0

spec_wave = []

for i in range ((y.shape[0]-window) // sift):
    wfunc = sp.signal.hamming(1024)
    tmp = y[i*sift : i*sift+window]
    tmp = tmp*wfunc
    spec = sp.fftpack.fft(tmp)
    spec_wave.append(spec)

plt.figure()
plt.figure(figsize=(15, 5))

seaborn.heatmap(np.array(spec_wave))
plt.show()
