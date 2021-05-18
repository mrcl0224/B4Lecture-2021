import numpy as np

'''
omega : angular frequency
'''

# numpy.sinc(x) = sinc(pi * x)


def LPF(imp_len, f1, fs, win):
    '''
    imp_len : impulse response's length
    f1 : cutoff frequency (default = 2000 Hz)
    fs : sampling frequency (default = 16000 Hz)
    win : window 
    '''
    omega = f1 / (fs / 2) * np.pi
    imp_num = np.arange(-imp_len // 2, imp_len // 2 + 1)
    # time_low_pass_filter[n] = (omega / pi) * sinc(omega * n)
    origin_filter = (omega / np.pi) * np.sinc(omega * imp_num / np.pi)

    my_filter = origin_filter * win

    return my_filter


def HPF(imp_len, f1, fs, win):
    '''
    imp_len : impulse response's length
    f1 : cutoff frequency (default = 2000 Hz)
    fs : sampling frequency (default = 16000 Hz)
    win : window
    '''
    omega = f1 / (fs / 2) * np.pi
    imp_num = np.arange(-imp_len // 2, imp_len // 2 + 1)
    # time_high_pass_filter[0] = 1 - omega / pi
    # time_high_pass_filter[n] = - (omega / pi) * sinc(omega * n)
    origin_filter = - (omega / np.pi) * np.sinc(omega * imp_num / np.pi)
    origin_filter[imp_len // 2] = 1 - omega / np.pi
    my_filter = origin_filter * win
    return my_filter


def BPF(imp_len, f1, f2, fs, win):
    '''
    imp_len : impulse response's length
    f1 : lower limit cutoff frequency (default = 2000 Hz)
    f2 : upper limit cutoff frequency (default = 4000 Hz)
    fs : sampling frequency (default = 16000 Hz)
    win : window
    '''
    omega1 = f1 / (fs / 2) * np.pi
    omega2 = f2 / (fs / 2) * np.pi
    imp_num = np.arange(-imp_len // 2, imp_len // 2 + 1)
    # time_band_pass_filter[n] =
    # (omega2 / pi) * sinc(omega2 * n) - (omega1 / pi) * sinc(omega2 / pi)
    origin_filter = (omega2 / np.pi) * np.sinc(omega2 * imp_num / np.pi) - \
                    (omega1 / np.pi) * np.sinc(omega1 * imp_num / np.pi)
    my_filter = origin_filter * win
    return my_filter


def BSF(imp_len, f1, f2, fs, win):
    '''
    imp_len : impulse response's length
    f1 : lower limit cutoff frequency (default = 2000 Hz)
    f2 : upper limit cutoff frequency (default = 4000 Hz)
    fs : sampling frequency (default = 16000 Hz)
    win : window
    '''
    omega1 = f1 / (fs / 2) * np.pi
    omega2 = f2 / (fs / 2) * np.pi
    imp_num = np.arange(-imp_len // 2, imp_len // 2 + 1)
    # time_high_stop_filter[0] = 1 - (omega2 - omega1) / pi
    # time_band_stop_filter[n] =
    # (omega2 / pi) * sinc(omega2 * n) - (omega1 / pi) * sinc(omega2 / pi)
    origin_filter = (omega1 / np.pi) * np.sinc(omega1 * imp_num / np.pi) - \
                    (omega2 / np.pi) * np.sinc(omega2 * imp_num / np.pi)
    origin_filter[imp_len // 2] = 1 - (omega2 - omega1) / np.pi
    my_filter = origin_filter * win
    return my_filter
