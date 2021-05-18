import numpy as np
import librosa
import librosa.display
from matplotlib import pyplot as plt
from scipy.fft import fft, ifft
from scipy import signal
import soundfile
import argparse

def stft(data_array, frame_size, overlap):
    result = np.array([])
    for frame_no in range(n_of_frame):
        left = frame_no * (frame_size - overlap)        # left end of frame
        right = left + frame_size                       # right end of frame
        frame = data_array[left: right]                # cutting one frame
        frame = frame * window_func                     # apply window function
        # append the result of fft
        result = np.append(result, fft(frame))
    result = result.reshape(n_of_frame, frame_size)
    return result


def istft(spec, win_len, overlap, wav_len):
    start = 0
    signal = np.zeros(wav_len)
    for slice in spec:
        ifft_result = np.fft.ifft(slice)
        signal[start: start+win_len] += np.real(ifft_result)
        start += (win_len-overlap)
    return signal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filename')
    parser.add_argument("-f", dest= "filename", help='Enter Filename', required=True)
    parser.add_argument("-fs", type = int, dest= "framesize", help='Enter Framesize', required=True)
    parser.add_argument("-os", type = int, dest= "overlap", help='Enter Overlap size', required=True)
    args = parser.parse_args()

    # sampling rate = 44100Hz
    wave_array, sr = librosa.load(
        args.filename, sr=44100)
    # size of one frame of stft
    frame_size = args.framesize
    # number of samples to overlap
    overlap = args.overlap
    # variable for window function
    window_func = signal.hamming(frame_size)
    # calculating the number of frames
    n_of_frame = len(wave_array) // (frame_size - overlap)
    p_wave_array = np.pad(wave_array, (0, overlap))           # zero padding
    # calculating total time
    time = wave_array / sr

    stft_result = stft(p_wave_array, frame_size, overlap)
    # transpose and convert to db
    db = librosa.amplitude_to_db((np.abs(stft_result)).T)
    # cutting out the symetrical part
    db = db[:len(db)//2]

    istft_result = istft(stft_result, frame_size, overlap, len(p_wave_array))
    istft_result = istft_result[:(len(wave_array) - frame_size)]

    # plot original waveform
    t = np.arange(0, len(wave_array)) / sr
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(wspace=0.2, hspace=0.6)
    plt.subplot(3, 1, 1)
    plt.plot(t, wave_array)
    plt.title('Original Signal')
    plt.xlabel('Time[s]')
    plt.ylabel('Magnitude')
    plt.xlim([0, len(p_wave_array)/sr])

    # plot spectogram of the result of stft
    plt.subplot(3, 1, 2)
    librosa.display.specshow(db, y_axis='linear', cmap='viridis')
    plt.colorbar()
    plt.title('Spectogram')

    # plot re-synthesized waveform
    p_istft_result = np.pad(
        istft_result, (0, len(t) - len(istft_result)))  # zero padding
    plt.subplot(3, 1, 3)
    plt.plot(t, p_istft_result)
    plt.title('Re-synthesized Signal')
    plt.xlabel('Time[s]')
    plt.ylabel('Magnitude')
    plt.xlim([0, len(p_wave_array)/sr])
    plt.show()

    soundfile.write('test.wav', p_istft_result.real, samplerate=sr)
