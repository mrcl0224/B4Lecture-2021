import numpy as np
import librosa
import matplotlib.pyplot as plt
import argparse
import soundfile
import os
import pdb


def specplot(mat, sr, frames):
    plt.xlabel("Time [sec]")
    plt.ylabel("Frequency [Hz]")
    plt.ylim(0, sr/2)    
    plt.imshow(mat, aspect="auto", extent=[0, frames / sr, 0, sr // 2])


def stft(data, win_len, win_type='hanning'):
    start = 0
    if win_type == 'hanning':
        window_n = np.hanning(win_len)
    elif win_type == 'hamming':
        window_n = np.hamming(win_len)

    slices = []
    while start < len(data):
        if len(data) - start >= win_len:
            slices.append(data[start: start+win_len])
        else:
            valid_signal = data[start: -1]
            slices.append(
                np.pad(valid_signal, (0, win_len-len(valid_signal)), 'constant'))
        start += win_len // 2
    slices = np.array(slices)
    # windowing
    windowed_slices = [window_n * slices[i] for i in range(len(slices[0:]))]

    spec = []
    for short_slice in windowed_slices:
        fft_result = np.fft.fft(short_slice)
        spec.append(fft_result)
    spec = np.array(spec).T
    abs_spec = np.abs(spec)
    return spec, abs_spec


def istft(spec, win_len, wav_len):
    start = 0
    signal = np.zeros(wav_len)
    for slice in spec.T:
        ifft_result = np.fft.ifft(slice)
        if start+win_len <= wav_len:
            signal[start: start+win_len] += np.real(ifft_result)
        else:
            signal[start: wav_len] += np.real(ifft_result[0: wav_len - start])
        start += win_len // 2
    return signal


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_file", type=str, default='arctic_b0340.wav',
                   help="something that can be helpful")
    p.add_argument("--win_len", type=int, default=1024,
                   help="window length for fft")
    p.add_argument("--sr", type=int, default=16000,
                   help="sample rate of input_file")
    p.add_argument('--window_type', type=str,
                   default="hanning", help="window type")
    p.add_argument("--output_tag", type=str,
                   default="", help="fig tag if isn't null")
    p.add_argument('--output_wav_toggle', type=int, default=0,
                   help="toggle for generation of wav file after ifft")

    args = p.parse_args()

    input_wav = args.input_file
    win_len = args.win_len

    # original signal
    y, sr = librosa.load(input_wav, sr=args.sr)

    # spectrogram
    spec, abs_spec = stft(y, win_len=win_len, win_type=args.window_type)
    abs_spec = abs_spec[win_len // 2: -1]  # 対称のため半分しか取らない
    mag_spec = 20 * np.log10(abs_spec)
    # new signal
    signal = istft(spec, win_len=win_len, wav_len=len(y))

    # plot
    # plt.subplots_adjust()

    # original signal
    plt.subplot(311)
    plt.plot(np.linspace(0, len(y)/sr, len(y)), y)
    plt.xlabel("Time [sec]")
    plt.ylabel("Amplitude")
    plt.xlim([0, len(y) / sr])
    plt.ylim([-1, 1])

    # spectrogram
    plt.subplot(312)
    specplot(mag_spec, sr=sr, frames=len(y))

    # new signal
    plt.subplot(313)
    plt.plot(np.linspace(0, len(signal)/sr, len(signal)), signal)
    plt.xlabel("Time [sec]")
    plt.ylabel("Amplitude")
    plt.xlim([0, len(y) / sr])
    plt.ylim([-1, 1])

    # plt.show()
    if args.output_tag == '':
        output_id = input_wav.split('/')[-1].split('.')[0] + '_win_type_' + \
            args.window_type + '_win_len_' + \
            str(args.win_len) + '_sr_' + str(args.sr)
    else:
        output_id = p.output_id
    
    plt.savefig(os.path.join('fig', output_id + ".png"))

    if args.output_wav_toggle:
        soundfile.write(os.path.join('wav',output_id + '.wav'), signal, samplerate=args.sr)
