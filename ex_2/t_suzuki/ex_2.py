import os
import argparse
import librosa
import soundfile as sf
import numpy as np
import spectrogram as sp
import graphics

# convolution function
def convolution(x, h):
    """
    x : array_like
        First input.
    h : array_like
        Second input.
    """

    x_len = len(x)
    h_len = len(h)
    y = np.zeros(x_len + h_len)

    for i in range(x_len):
        y[i:i+h_len] += x[i] * h

    return y

# sinc function
def sinc(x):
    return np.where(x == 0, 1, np.sin(x) / x)

# low path filter
def LPF(sr, fc, N):
    """
    sr : sampling rate
    fc : cutoff frequency
    N  : the number of filter coefficients
    """

    pi = np.pi
    wc = 2 * pi * fc 
    wcn = wc / sr

    arange = np. arange(-N//2, N//2+1)
    h = wcn * sinc(wcn * arange) / pi
    window_func = np.hamming(N+1)

    return h * window_func


def main():
    # argment settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input filename")
    parser.add_argument("--freq_out", type=str, default="freq_resuponse.png", help="frequency response figure name")
    parser.add_argument("--spec_out", type=str, default="spectrogram.png", help="spectrogram figure name")
    parser.add_argument("--audio_out", type=str, default="filterd_sound.wav", help="output audio file name")
    parser.add_argument("--fc", type=int, default=2000, help="cut off frequency")
    parser.add_argument("--N", type=int, default=100, help="the number of filter coefficients")
    parser.add_argument("--frame", type=int, default=1024, help="spectrogram frame size")
    parser.add_argument("--overlap", type=float, default=0.5, help="overlap rate")
    args = parser.parse_args()

    if not os.path.exists('./out'):
        os.makedirs('./out')

    # load audio file
    y, sr = librosa.load(args.input)
    time = np.arange(0, len(y)) / sr

    # make filter
    lpf = LPF(sr, args.fc, args.N)
    lpf_fft = np.fft.fft(lpf)
    lpf_amp = 20 * np.log10(np.abs(lpf_fft))
    phase = np.unwrap(np.angle(lpf_fft)) * 180 / np.pi
    freq = np.arange(0, sr/2, (sr//2)/(args.N//2 + 1))

    # convolution
    y_filtered = convolution(y, lpf)
    # spectrogram
    original_spec = sp.spec(y, args.frame, args.frame * args.overlap)
    filtered_spec = sp.spec(y_filtered, args.frame, args.frame * args.overlap)

    # make figure
    freq_fig = graphics.draw_freq_res(freq, lpf_amp, phase, args.N)
    spec_fig = graphics.draw_spec(original_spec, filtered_spec, sr, time[-1])

    # save figure
    freq_fig.savefig(f'./out/{args.freq_out}')
    spec_fig.savefig(f'./out/{args.spec_out}')

    # save filterd sound
    sf.write(f'./out/{args.audio_out}', y_filtered, sr)

if __name__ == "__main__":
    main()
