import argparse

import librosa
import librosa.display
import numpy as np
from matplotlib import pyplot as plt
import soundfile

import ex_1.a_miyashita.main as ex1


def convolution(input, filter):
    """
    Args:
        input   (ndarray, shape=(n,))
        filter  (ndarray, shape=(m,))

    Returns:
        result  (ndarray, shape=(n+m,))
    """

    result = np.zeros(input.size + filter.size)

    for i in range(input.size):
        # convolution
        result[i : i + filter.size] = result[i : i + filter.size] + input[i] * filter

    return result


class BSF:
    def __init__(self, fmin, fmax, win_size, sr):
        """
        Args:
            fmin    (int):the lowest frequency of the band stopped
            fmax    (int):the highest frequency of the band stopped
            win_size(int):sizeof window applied to filter
            sr      (int):sampling rate of the signal
        """

        # normalize
        wmin = 2.0 * fmin / sr
        wmax = 2.0 * fmax / sr

        ite = np.arange(
            -(win_size // 2),
            (win_size + 1) // 2,
        )

        # culculate -(BPF)
        self.filter = wmin * np.sinc(wmin * ite) - wmax * np.sinc(wmax * ite)

        # add APF
        self.filter[win_size // 2] += 1.0

        # apply window function
        self.filter *= np.hamming(win_size)

        # internal status remembering past (win_size) inputs
        self.status = np.zeros(win_size)

        self.win_size = win_size

    def __call__(self, inputs=None):
        """
        Apply filter to inputs.

        Args:
            inputs  ((ndarray, shape=(n,)) or float)

        Returns:
            If inputs is None, self.status is returned. Then self.status is reset.
            Otherwize, self.filter is convoluved to inputs. The latter (win_size) of results are
            remembered as self.status and the rest (inputs.size) are returned.
        """

        if inputs is None:
            status = self.status

            # reset status
            self.status = np.zeros(self.win_size)

            return status
        else:
            # to convert scalar to 0-dim array
            inputs = np.array(inputs)

            # convolution
            conv = convolution(inputs, self.filter)

            # add effects of past inputs
            conv[: self.win_size] += self.status

            # update status
            self.status = conv[inputs.size :]

            return conv[: inputs.size]


if __name__ == "__main__":
    # process args
    parser = argparse.ArgumentParser(description="apply band-stop-filter to sound data")
    parser.add_argument("sc", type=str, help="input filename with extension .wav")
    parser.add_argument("dst", type=str, help="output filename with extension .wav")
    parser.add_argument(
        "fmin", type=int, help="the lowest frequency in the band you stop [Hz]"
    )
    parser.add_argument(
        "fmax", type=int, help="the highest frequency in the band you stop [Hz]"
    )
    parser.add_argument("win_size", type=int, help="size of window applied to filter")
    args = parser.parse_args()

    # load
    original_signal, rate = soundfile.read(args.sc)

    # create band-stop-filter
    bsf = BSF(args.fmin, args.fmax, args.win_size, rate)

    # analize filter
    freq = np.fft.rfft(bsf.filter, rate)
    amp = np.abs(freq)
    phase = np.unwrap(np.angle(freq))

    # apply filter
    filtered_signal = bsf(original_signal)
    filtered_signal = np.concatenate([filtered_signal, bsf()])
    # when input signal is given in sequence, you can use like this:
    # filtered_signal[ite] = bsf(original_signal[ite])

    n_fft = 512

    # stft
    spectrogram1 = ex1.stft(original_signal, n_fft)
    spectrogram2 = ex1.stft(filtered_signal, n_fft)

    # convert to db
    spectrogram_db1 = librosa.amplitude_to_db(np.abs(spectrogram1))
    spectrogram_db2 = librosa.amplitude_to_db(np.abs(spectrogram2))

    # plot
    plt.rcParams["figure.figsize"] = (12, 15)
    fig, ax = plt.subplots(4, 1)
    fig.subplots_adjust(hspace=0.5)

    ax[0].plot(amp)
    ax[0].set(title="Filter amplitude", xlabel="Frequency[Hz]", ylabel="Amplitude")

    ax[1].plot(phase)
    ax[1].set(title="Filter phase", xlabel="Frequency[Hz]", ylabel="Phase[rad]")

    img = librosa.display.specshow(
        spectrogram_db1,
        sr=rate,
        hop_length=n_fft // 2,
        x_axis="time",
        y_axis="linear",
        ax=ax[2],
        cmap="rainbow",
    )
    ax[2].set(title="Original", xlabel="Time[sec]", ylabel="Frequency[Hz]")
    fig.colorbar(img, ax=ax[2])

    img = librosa.display.specshow(
        spectrogram_db2,
        sr=rate,
        hop_length=n_fft // 2,
        x_axis="time",
        y_axis="linear",
        ax=ax[3],
        cmap="rainbow",
    )
    ax[3].set(title="Filtered", xlabel="Time[sec]", ylabel="Frequency[Hz]")
    fig.colorbar(img, ax=ax[3])

    ax_pos_0 = ax[0].get_position()
    ax_pos_1 = ax[1].get_position()
    ax_pos_2 = ax[2].get_position()
    ax_pos_3 = ax[3].get_position()
    ax[0].set_position([ax_pos_0.x0, ax_pos_0.y0, ax_pos_2.width, ax_pos_2.height])
    ax[1].set_position([ax_pos_1.x0, ax_pos_1.y0, ax_pos_2.width, ax_pos_2.height])

    plt.savefig("result")

    # save filtered sound
    soundfile.write(args.dst, filtered_signal, samplerate=rate)
