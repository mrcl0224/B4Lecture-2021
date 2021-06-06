import librosa
import librosa.display
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import soundfile as sf

def ACF(y):

    N = len(y)
    out = np.zeros(N - 1)

    for i in range(N - 1):

        for j in range(N - i):

            out[i] += y[j] * y[j + i]

    return out

def ACT(r, sr):

    if np.argmax(r) == 0:
        print("0 divide!!!!!!")
        return 0
    else:

        fo = sr / np.argmax(r)

        return fo

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

if __name__ == "__main__":

    # データの取り込み。
    fname = "Mic_test.wav"
    y, sr = librosa.load(fname, sr=44100)

    time = np.arange(0, len(y)/sr, 1/sr)

    window_width = 1024
    shift_size = 512
    wfunc = sp.signal.hamming(window_width)

    #自己相関法を用いる。
    fo_result = np.zeros((y.shape[0] - window_width) // shift_size)

    for i in range((y.shape[0] - window_width) // shift_size):

        tmp = y[i * shift_size : i * shift_size + window_width]
        tmp = tmp * wfunc

        r = ACF(tmp)
        m_min = np.argmin(r)
        fo = ACT(r[m_min:], sr)
        fo_result[i] += fo

    #print(fo_result)

    fig = plt.figure(figsize = (10,4))
    ax = fig.add_subplot(111, title = 'Mel Scale')
    ax.plot(fo_result)
    fig.show()

    #ケプストラム法を用いる。

    #これで対数パワースペクトルまで変換できる
    y_fft = STFT(y, window_width, shift_size)

    y_ceps = np.real(sp.fftpack.ifft(y_fft))

    img_spec = plt.figure(figsize=(15, 5))
    plt.imshow(
        y_fft,
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

    #print(y_ceps)

    quefrency = time - min(time)

    fig = plt.figure(figsize = (10,4))
    ax = fig.add_subplot(111, title = 'Cepstrum')
    ax.plot(y_ceps[500], color = '#FF0000')
    ax.set_xlabel('Quefency[ms]')
    ax.set_ylabel('Cepstrum[Hz]')
    plt.show()
