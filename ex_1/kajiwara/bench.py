import time

import scipy as sp
import numpy as np
import librosa
import matplotlib.pyplot as plt
from datetime import datetime

from main import get_spectrogram

TIME_TEMPLATE = '%Y%m%d%H%M%S'


def calc_process_time(args, func):
    start = time.perf_counter()
    func(**args)
    end = time.perf_counter()

    return end - start


def librosa_spectrogram(data):
    cv = librosa.stft(data)
    # s, phase = librosa.magphase(cv)
    db = librosa.amplitude_to_db(np.abs(cv))

    return db


def main():
    length = [
        22050*1,
        22050*10,
        22050*50,
        22050*100,
        22050*150,
        22050*500,
        22050*1000,
        22050*1500,
    ]
    result = []

    for l in length:
        print(l)
        fake_data = 2 * np.random.rand(l) - 1

        res = {
            'scipy': [],
            'librosa': [],
            'mine_normal': [],
            'mine_faster': [],
        }

        for i in range(10):
            # scipy
            sp_res = calc_process_time(
                {'x': fake_data, 'mode': 'complex'}, sp.signal.spectrogram)
            res['scipy'].append(sp_res)

            # librosa
            lib_res = calc_process_time(
                {'data': fake_data}, librosa_spectrogram)
            res['librosa'].append(lib_res)

            # mine(normal)
            my_res = calc_process_time(
                {'wave_data': fake_data, 'mode': 'normal'}, get_spectrogram)
            res['mine_normal'].append(my_res)

            # mine(faster)
            my_res = calc_process_time(
                {'wave_data': fake_data, 'mode': 'faster'}, get_spectrogram)
            res['mine_faster'].append(my_res)

        result.append(res)

    aves = {
        'scipy': [],
        'librosa': [],
        'mine_normal': [],
        'mine_faster': [],
    }
    for i, res in enumerate(result):
        print('-'*10)
        for k, v in res.items():
            print(k, sum(v)/len(v))
            aves[k].append(sum(v)/len(v))
    print('-'*10)

    p_scipy = plt.plot(length, aves['scipy'], marker="o", label='scipy')
    p_librosa = plt.plot(length, aves['librosa'], marker="o", label='librosa')
    p_normal = plt.plot(
        length, aves['mine_normal'], marker="o", label='normal')
    p_faster = plt.plot(
        length, aves['mine_faster'], marker="o", label='faster')
    plt.legend(loc=2)
    timestamp = datetime.now().strftime(TIME_TEMPLATE)
    plt.savefig('bench-res-{}.png'.format(timestamp))


main()
