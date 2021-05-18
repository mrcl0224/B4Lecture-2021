import math

import numpy as np
import scipy as sp
import librosa


def sinc(x):
    if x == 0.0:
        return 1.0
    else:
        return np.sin(x) / x

    return sinc


def low_pass_filter():
    # TODO: LPFの実装
    pass


def high_pass_filter():
    # TODO: HPFの実装
    pass


def band_pass_filter(sr, h_cutoff, l_cutoff, n_flt):
    '''
    this function generate band-pass filter

    Parameters
    ----------
    sr: int
        sampling rate
    h_cutoff, l_cutoff:
        edge frequency (l_cutoff < h_cutoff)
    n_flt: int
        number of filter coefficients

    Returns
    -------
    filter: ndarray
    '''

    assert h_cutoff > l_cutoff

    # generate frequency characteristic
    # fc = np.zeros(sr)
    # fc[l_cutoff:h_cutoff] = 1
    # fc[-h_cutoff:-l_cutoff] = 1

    # get inpluse response
    bpf = np.zeros(n_flt+1)
    f1 = l_cutoff / sr
    f2 = h_cutoff / sr
    for i, n in enumerate(range(-n_flt//2, n_flt//2 + 1)):
        bpf[i] = (2 * f2 * sinc(2 * math.pi * f2 * n) -
                  2 * f1 * sinc(2 * math.pi * f1 * n))

    # window function
    window = np.hamming(n_flt+1)
    bpf *= window

    return bpf


def band_stop_filter():
    # TODO: BEFの実装
    pass
