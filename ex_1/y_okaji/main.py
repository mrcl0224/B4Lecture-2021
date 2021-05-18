import sys
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf


def get_window_index(window_size, hop_length, last_index):
    """[summary]

    Args:
        window_size ([int]): [size of window]
        hop_length ([int]): [size of stride]
        last_index ([int]): [last index of entire data]

    Returns:
        frame_start [np.ndarray]: [start index of each frame]
        frame_end [np.ndarray]: [end index of each frame]
    """
    
    frame_start = np.array([]).astype(np.int64)
    frame_end = np.array([]).astype(np.int64)
    
    start = 0
    while True:
        frame_start = np.append(frame_start, start)
        end = min(start+window_size, last_index)
        frame_end = np.append(frame_end, end)
        if end == last_index:
            break
        start += hop_length
    
    frame_num = int(len(frame_start))
    
    return frame_start, frame_end, frame_num

def stft(data, n_fft=1024, hop_length=512):
    """[summary]

    Args:
        data ([np.ndarray]): [shape=(n, )] real-valued the input signal (audio time series)
        n_fft (int, optional): [FFT window]. Defaults to 1024.
        hop_length (int, optional): [the size of stride in FFT]. Defaults to 512.
    """
    
    hamming = np.hamming(n_fft)                                         # make hamming window 

    frame_start, frame_end, frame_num = get_window_index(n_fft, hop_length, len(data))
    stft_mat = np.zeros((n_fft, frame_num), dtype=complex)
    for i, (start, end) in enumerate(zip(frame_start, frame_end)):
        cut_data = data[start:end]                                      # cut data for FFT
        if len(cut_data) != n_fft:                                      # zero padding
            cut_data = np.append(cut_data, np.zeros(n_fft-len(cut_data)))        
        data_win = cut_data * hamming                                   # apply hamming window to audio data
        data_trans = np.fft.fft(data_win)                               # FFT
        stft_mat[:, i] = data_trans                                     # append the result of FFT 
        
    return stft_mat

def istft(spec, n_fft=1024, hop_length=512):
    """[summary]

    Args:
        spec ([type]): [output ndarray from stft function]
        n_fft (int, optional): [the size of FFT window]. Defaults to 1024.
        hop_length (int, optional): [the size of stride in FFT]. Defaults to 512.
    """
    
    frame_num = spec.shape[1]
    reconst_audio = np.zeros(frame_num * hop_length + n_fft, dtype=complex)
    
    for i in range(frame_num):
        start = i * hop_length
        end = start + n_fft
        ifft_result = np.fft.ifft(spec[:, i])
        reconst_audio[start:end] += ifft_result
    
    return reconst_audio

if __name__ == "__main__":
    
    # make save directory
    save_dir = "./figure"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    # import data
    wav_path = "./data/onso_balance_01_okaji.wav" 
    data, sr = librosa.load(wav_path, sr=None)
    
    # parameter
    n_fft = 1024
    hop_length = 512
    
    # stft
    stft_result = stft(data, n_fft=n_fft, hop_length=hop_length)
    spec = np.abs(stft_result[:int(n_fft/2), :])
    log_spec = 20 * np.log10(spec + sys.float_info.epsilon)
    
    # istft
    reconst = istft(stft_result, n_fft=n_fft, hop_length=hop_length).real
    reconst = reconst[:len(data)]
    
    # output the reconst audio wav
    sf.write("./data/reconst_audio.wav", reconst, sr)
    
    # compare the raw audio to reconst data
    rmse = np.sqrt(np.mean((data - reconst)**2))
    print(rmse)
    
    # draw three figures
    time = np.arange(0, len(data)/sr, 1/sr)
    frequency = np.fft.fftfreq(n_fft, d=1/sr)[:int(n_fft/2)]
    plt.figure(figsize=(8,6))
    plt.subplots_adjust(hspace=0.8)
    ## original audio dat
    plt.subplot(3, 1, 1)
    plt.plot(time, data)
    plt.title("original")
    plt.xlabel("time [sec]")
    plt.ylabel("amplitude")
    
    ## spectrogram
    plt.subplot(3, 1, 2)
    librosa.display.specshow(log_spec,
                             sr=sr, 
                             hop_length=hop_length, 
                             x_axis="time",
                             y_axis="log",
                             cmap="magma")
    plt.colorbar(format='%02.0f dB')
    plt.title("Spectrogram")
    plt.xlabel("time")
    plt.ylabel("frequency")
    
    ## reconstruction data
    
    plt.subplot(3, 1, 3)
    plt.plot(time, reconst)
    plt.title("Inversed")
    plt.xlabel("time [sec]")
    plt.ylabel("amplitude")
    
    plt.savefig(save_dir+"/result.jpg")