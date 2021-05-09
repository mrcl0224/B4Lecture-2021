import numpy as np
import scipy as sp
import librosa
import seaborn
import math

fname = "Sing_Data_1.wav"
y, sr = librosa.load(fname, sr = 44100)

import matplotlib.pyplot as plt
import librosa.display

#print(y.shape)

#元の波形を表示。
fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(ylabel='Amplitude')
librosa.display.waveplot(y, sr)
ax1.set(xlabel = "Time[s]")
fig.suptitle("Original Wave")
plt.show()

#窓関数の枠は1024、シフト幅は512(1024の半分)で設定。
window = 1024
shift = 512
i = 0

#変換後の波の情報を記録するためのリスト。
spec_wave = []

#フーリエ変換。
for i in range ((y.shape[0]-window) // shift):
    wfunc = sp.signal.hamming(1024)
    tmp = y[i*shift : i*shift+window]
    tmp = tmp*wfunc
    spec = sp.fftpack.fft(tmp)
    spec_wave.append(spec)

#print(np.array(spec_wave).shape)

#フーリエ変換の結果には虚数が含まれるため、絶対値を取る。
#転置することで、縦軸が周波数、横軸が時間となる。
#振幅をデシベルに変換。y = 20log(x) (参照：https://www.onosokki.co.jp/HP-WK/c_support/newreport/decibel/dB.pdf)
result_spec = 20.0*np.log(np.array(np.abs(spec_wave).T))[:512]

freq = np.fft.fftfreq(result_spec.shape[0], 1.0/sr)[:512]
time = np.arange(result_spec.shape[1]) * shift * (1.0/sr)

#描画。
img_spec = plt.figure(figsize=(15, 5))
#img_spec.add_subplot(xlim = (0.0, time.max()), xticks = range(0, math.floor(time.max()), 5), ylim = (0, freq.max()), yticks = [0, 1000,10000,20000])
plt.imshow(result_spec, vmin=-100.0, vmax=40, cmap = "rainbow", origin = "lower",extent = [0, np.max(time), 0, sr/2], aspect = "auto")
plt.title("Spectrogram")
plt.ylabel("Frequency[Hz]")
plt.xlabel("Time[s]")
#seaborn.heatmap(result_spec, vmin=-100.0, vmax=40, cmap = "rainbow", xticks=time, yticks=freq)
plt.show()

#もう一度for文を使うため初期化。
i = 0
j = 0

#逆フーリエ変換の結果を保存するためのndarrayを作成。
re_wave = np.zeros(np.array(spec_wave).shape[0]*shift + window)

#逆フーリエ変換。逆フーリエ変換の結果にも虚数が含まれるため、絶対値を取る。
#シフト幅でずらして足していたので、逆に元に戻るようにずらしてた分が重なるようにする。
for i in range(np.array(spec_wave).shape[0]):
    re = sp.fftpack.ifft(spec_wave[i])
    for j in range(np.array(re).shape[0]):
        re_wave[i*shift + j] += np.abs(re[j])

print(re_wave.shape)

result_re = np.abs(np.array(re_wave))

#描画。
fig = plt.figure(figsize=(15, 5))
ax2 = fig.add_subplot(ylabel='Amplitude')
librosa.display.waveplot(result_re, sr)
ax2.set(xlabel = "Time[s]")
fig.suptitle("Wave after IFFT")
plt.show()

#音声ファイルとして書き出してみる。
import soundfile as sf
sf.write("test.wav", result_re, sr, subtype="PCM_16")
