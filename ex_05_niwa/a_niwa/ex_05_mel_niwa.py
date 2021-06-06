import sys
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import random
from mpl_toolkits.mplot3d import Axes3D
import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy as sp
import soundfile as sf

def STFT(y, window_width, shift_size):
    """
    フーリエ変換を行う。

    Parameters
    ----------
    y : ndarray
        フーリエ変換を行う配列、
    window_with : int
        フーリエ変換を実行できるようにするための窓枠。
    shift_size : int
        切り取りの開始位置。

    Returns
    -------
    result_spec : ndarray
        フーリエ変換を行い、周波数領域で表された波形。
    """
    # 変換後の波の情報を記録するためのリスト。
    spec = np.zeros(
        ((y.shape[0] - window_width) // shift_size, window_width), dtype=np.complex128
    )
    wfunc = sp.signal.hamming(window_width)
    for i in range((y.shape[0] - window_width) // shift_size):
        tmp = y[i * shift_size : i * shift_size + window_width]
        tmp = tmp * wfunc
        spec[i] += sp.fftpack.fft(tmp)
    # フーリエ変換の結果には虚数が含まれるため、絶対値を取る。
    # 転置することで、縦軸が周波数、横軸が時間となる。
    # 振幅をデシベルに変換。y = 20log(x) (参照：https://www.onosokki.co.jp/HP-WK/c_support/newreport/decibel/dB.pdf)
    result_spec = 20.0 * np.log(np.array(np.abs(spec).T))[:512]
    return result_spec

def freq2mel(freq, mo = 2595/math.log(10), fo = 700):

    mel = mo * math.log((freq / fo) + 1)

    return mel

def mel2freq(mel, mo = 2595/math.log(10), fo = 700):

    freq = fo * (np.exp(mel/mo) - 1)

    return freq

def mel_filter_bank(sr):

    pass

def DCT(y):

    y_DCT = np.zeros_like(y)
    for i in range(y_DCT.shape[0]):
        for j in range(y.shape[0]):
            y_DCT[j] += math.sqrt(2/(y.shape[0])) * y[i] * math.cos(((2*i + 1)*j*math.pi)/(2*y.shape[0]))

    y_DCT[0] = y_DCT[0]*(1/math.sqrt(2))

    return y_DCT

if __name__ == "__main__":

    fname = "Sing_Data_1.wav"
    y, sr = librosa.load(fname, sr=44100)

    time = np.arange(0, len(y)/sr, 1/sr)

    window_width = 1024
    shift_size = 512
    wfunc = sp.signal.hamming(window_width)

    #関数内でパワースペクトルに変換されるように実装されている
    y_FFT = STFT(y, window_width, shift_size)

    mfb_size = 64

    max_mel = freq2mel(sr / 2)

    flt_size = 512

    mel_x = np.linspace(0, max_mel, mfb_size + 2)
    freq_x = np.round(mel2freq(mel_x) * flt_size / (sr/2)).astype(int)

    filter_bank = np.zeros((mfb_size, flt_size))
    for i in range(mfb_size):
        filter_bank[i, freq_x[i]:freq_x[i+1]] = np.linspace(0, 1, freq_x[i+1] - freq_x[i])
        filter_bank[i, freq_x[i+1]:freq_x[i+2]] = np.linspace(1, 0, freq_x[i+2] - freq_x[i+1])

    y_mel = np.dot(filter_bank, y_FFT)

    dim = 12
    y_DCT = DCT(y_mel)[1:dim+1]

    l = 4
    delta_MFCC = np.zeros((y_DCT.shape[0],y_DCT.shape[1]-(2*l)))

    k_pow_sum = 0

    for j in range(-l,l+1):
        k_pow_sum += pow(j, 2)

    for i in range(delta_MFCC.shape[1]):
        for j in range(-l,l+1):
            delta_MFCC[i] += (j * (y_DCT[l:y_DCT.shape[1]-l])[i + l + j]) / k_pow_sum

    print(delta_MFCC[i])


    # フィルタ適用前の波形を表示。
    img_spec = plt.figure(figsize=(15, 5))
    plt.imshow(
        y_FFT,
        vmin=-100.0,
        vmax=40,
        cmap="rainbow",
        origin="lower",
        extent=[0, np.max(time), 0, sr / 2],
        aspect="auto",
    )
    plt.title("Before Filtering")
    plt.ylabel("Frequency[Hz]")
    plt.xlabel("Time[s]")
    plt.show()

    # フィルタ適用後の波形表示。
    img_spec = plt.figure(figsize=(15, 5))
    plt.imshow(
        y_DCT,
        vmin=np.min(y_DCT),
        vmax=np.max(y_DCT),
        cmap="rainbow",
        origin="lower",
        extent=[0, np.max(time), 1, 12],
        aspect="auto",
    )
    plt.title("After Filtering")
    plt.ylabel("MCFF")
    plt.xlabel("Time[s]")
    plt.show()

    #MFCC_wav = (sp.fftpack.ifft(DCT(y_mel)).real)

    #sf.write("test.wav", MFCC_wav, sr, subtype="PCM_16")
