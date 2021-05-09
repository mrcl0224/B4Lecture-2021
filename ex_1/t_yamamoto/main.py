# include flake8, black

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import warnings

# ignore warnings
warnings.filterwarnings("ignore")


def stft(y, hop=0.5, win_length=1024):
    """
    Compute the Short Time Fourier Transform (STFT).

    Parameters:
        y : np.ndarray, real-valued
            Time series of measurement values.
        hop : float [scaler], optional
            Hop (Overlap) size.
        win_length : int [scaler], optional
            Window size.

    Return:
        F : np.ndarray
            Complex-valued matrix of short-term Fourier transform coefficients.
    """

    hop_length = int(win_length * hop)
    ynum = y.shape[0]
    window = np.hamming(win_length)

    F = []
    for i in range(int((ynum - hop_length) / hop_length)):
        tmp = y[i * hop_length : i * hop_length + win_length]
        # multiplied by window finction
        tmp = tmp * window
        # Fast Fourier Transform (FFT)
        tmp = np.fft.rfft(tmp)
        F.append(tmp)

    # (frame, freq) -> (freq, frame)
    F = np.transpose(F)
    return F


def istft(F, hop=0.5, win_length=1024):
    """
    Compute the Short Time Fourier Transform (STFT).

    Parameters:
        F : np.ndarray
            Complex-valued matrix of short-term Fourier transform coefficients.
        hop : float [scaler], optional
            Hop (Overlap) size.
        win_length : int [scaler], optional
            Window size.

    Return:
        y : np.ndarray
            Time domain signal.
    """

    hop_length = int(win_length * hop)
    window = np.hamming(win_length)
    # (freq, frame) -> (frame, freq)
    F = np.transpose(F)
    # Inverse Fast Fourier Transform (IFFT)
    tmp = np.fft.irfft(F)
    # divided by window function
    tmp = tmp / window
    # remove overlap
    tmp = tmp[:, :hop_length]
    y = tmp.reshape(-1)

    return y


def main():
    # load audio file
    # get current working directory
    dir = os.path.dirname(__file__) + "/"
    # dir = "C:/Users/yamam/Desktop/lab/2021/B4Lecture-2021/ex_1/t_yamamoto/"
    audio_path = dir + "recording_b4lec_ex1.wav"
    wav, sr = librosa.load(audio_path, mono=True)

    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    plt.subplots_adjust(hspace=0.6)

    # draw original signal
    librosa.display.waveplot(wav, sr=sr, color="g", ax=ax[0])
    ax[0].set(title="Original signal", xlabel=None, ylabel="Magnitude")

    # parameter
    hop = 0.5
    win_length = 1024
    hop_length = int(win_length * hop)

    # STFT
    amp = stft(wav, hop=hop, win_length=win_length)
    # convert an amplitude spectrogram to dB-scaled spectrogram
    db = librosa.amplitude_to_db(np.abs(amp))
    # db = librosa.amplitude_to_db(np.abs(librosa.stft(wav)), ref=np.max)
    # draw spectrogram (log scale)
    img = librosa.display.specshow(
        db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="log",
        ax=ax[1],
        cmap="plasma",
    )
    ax[1].set(title="Spectrogram", xlabel=None, ylabel="Frequency [Hz]")
    ax[1].set_yticks([0, 128, 512, 2048, 8192])
    fig.colorbar(img, aspect=10, pad=0.01, extend="both", ax=ax[1], format="%+2.f dB")

    # inverse-STFT
    inv_wav = istft(amp, hop=hop, win_length=win_length)
    # draw re-synthesized signal
    librosa.display.waveplot(inv_wav, sr=sr, color="g", ax=ax[2])
    ax[2].set(title="Re-synthesized signal", xlabel="Time [s]", ylabel="Magnitude")

    # graph adjustment
    ax_pos_0 = ax[0].get_position()
    ax_pos_1 = ax[1].get_position()
    ax_pos_2 = ax[2].get_position()
    ax[0].set_position([ax_pos_0.x0, ax_pos_0.y0, ax_pos_1.width, ax_pos_1.height])
    ax[2].set_position([ax_pos_2.x0, ax_pos_2.y0, ax_pos_1.width, ax_pos_1.height])
    # fig.tight_layout()
    fig.align_labels()

    # save and show figure of result
    plt.savefig(dir + "ex1_result.png")
    plt.show()


if __name__ == "__main__":
    main()
