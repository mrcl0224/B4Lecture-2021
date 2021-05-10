import librosa
import librosa.display
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import soundfile as sf


def my_conv(x, y):
    """
    畳み込み積分を行う。

    Parameters
    ----------
    x, y : ndarray
        畳み込みを行う配列2つ

    Returns
    -------
    z : ndarray
        畳み込みの結果を返す
    """
    x_len = len(x)
    y_len = len(y)

    z = np.zeros(x_len + y_len - 1)

    for i in range(y_len):
        z[i : i + x_len] += x * y[i]

    return z


def sinc(x):
    """
    sinc関数を定義。

    Parameters
    ----------
    x : float
        入力値。

    Returns
    -------
    answer : float
        出力値。
    """
    if x == 0:
        answer = 1
    else:
        answer = math.sin(x) / x
    return answer


def LowPassFilter(LPfreq, LP_window_width, sr):
    """
    ローパスフィルタを定義する。

    Parameters
    ----------
    LPfreq : float
        ローパスの上限値を設定する。
    LP_window_width : int
        FIRフィルタで設計しているので、窓枠の設定。
    sr : int
        音源のサンプルレート。

    Returns
    -------
    LP : ndarray
        時間領域でのローパスフィルタを返す。
    """
    # ローパスフィルタは有限長で作成する。
    # 有限長分の空のndarrayを作成。
    b = np.zeros(2 * LP_window_width + 1)
    for i in range(-LP_window_width, LP_window_width + 1):
        LPwave = 2 * (LPfreq / sr) * sinc(2 * math.pi * (LPfreq / sr) * i)
        b[i + LP_window_width] += LPwave
    wfunc = sp.signal.hamming(2 * LP_window_width + 1)
    LP = wfunc * b
    return LP


def SFTF(y, window_width, shift_size):
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
    fname = "Sing_Data_1.wav"
    y, sr = librosa.load(fname, sr=44100)
    # print(y)

    # ローパスフィルタの生成。
    LPfreq = 2000  # 上限の周波数を決定。今回は2000Hzで切る。
    LP_window_width = 512
    Filter = LowPassFilter(LPfreq, LP_window_width, sr)
    # print(Filter)

    # 振幅特性と周波数特性の表示のためにフーリエ変換する。
    Filter_FFT = np.abs(sp.fftpack.fft(Filter))[: (Filter.shape[0]) // 2]
    # 振幅特性描画
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111, xlabel="Frequency[Hz]", ylabel="Amplitude[dB]")
    plt.title("Amplitude response")
    plt.xticks(np.linspace(0, LP_window_width, 11), np.linspace(0, sr / 2, 11))
    ax.grid()
    ax.plot(Filter_FFT)
    plt.show()

    # 周波数特性描画
    # Filter_FFT_Phase = np.zeros((Filter.shape[0])//2)
    tmp = sp.fftpack.fft(Filter)
    # for i in range((Filter.shape[0])//2):
    #    Filter_FFT_Phase[i] += np.angle(tmp[i])
    Filter_FFT_Phase = (
        np.unwrap(np.angle(tmp)[: (Filter.shape[0]) // 2]) * 180 / math.pi
    )
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111, xlabel="Frequency[Hz]", ylabel="Phase")
    ax.plot(Filter_FFT_Phase)
    plt.title("frequency response")
    plt.xticks(np.linspace(0, LP_window_width, 11), np.linspace(0, sr / 2, 11))
    ax.grid()
    plt.show()

    # 畳み込み
    result = my_conv(y, Filter)
    # print(result)

    # スペクトログラムのためにフーリエ変換。
    window_width = 1024
    shift_size = 512

    y_FFT = SFTF(y, window_width, shift_size)
    result_FFT = SFTF(result, window_width, shift_size)

    y_time = np.arange(y_FFT.shape[1]) * shift_size * (1.0 / sr)

    # フィルタ適用前の波形を表示。
    img_spec = plt.figure(figsize=(15, 5))
    plt.imshow(
        y_FFT,
        vmin=-100.0,
        vmax=40,
        cmap="rainbow",
        origin="lower",
        extent=[0, np.max(y_time), 0, sr / 2],
        aspect="auto",
    )
    plt.title("Before Filtering")
    plt.ylabel("Frequency[Hz]")
    plt.xlabel("Time[s]")
    plt.show()

    result_time = np.arange(result_FFT.shape[1]) * shift_size * (1.0 / sr)

    # フィルタ適用後の波形表示。
    img_spec = plt.figure(figsize=(15, 5))
    plt.imshow(
        result_FFT,
        vmin=-100.0,
        vmax=40,
        cmap="rainbow",
        origin="lower",
        extent=[0, np.max(result_time), 0, sr / 2],
        aspect="auto",
    )
    plt.title("After Filtering")
    plt.ylabel("Frequency[Hz]")
    plt.xlabel("Time[s]")
    plt.show()

    # 音声ファイルとして書き出してみる。
    sf.write("test.wav", result, sr, subtype="PCM_16")
