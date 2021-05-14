#!/usr/bin/env python3

import numpy as np
import os
import sys
os.chdir(sys.path[0])

import wfdb
from utils import qrs_detect, comp_cosEn, save_dict
from tensorflow.keras.models import load_model
from scipy.signal import ellip, ellipord, filtfilt, lfilter, butter


def load_data(sample_path):
    sig, fields = wfdb.rdsamp(sample_path)
    length = len(sig)
    fs = fields['fs']

    return sig, length, fs


def ngrams_rr(data, length):
    grams = []
    for i in range(0, length-12, 12):
        grams.append(data[i: i+12])
    return grams


def sig_filter(signal, fs):
    rp, rs = 1, 20
    wn = 2 * np.array([0.5, 35]) / fs  # Wn = 截止频率 / 信号频率，信号频率=采样率的一半
    n = 5
    b, a = ellip(n, rp, rs, wn, btype='bandpass')
    return filtfilt(b, a, signal, axis=0, method='gust')


def load_test_data(signal, sample, interval=1):
    test_x = np.zeros(shape=(1, length, 2))

    start = 0
    while start + interval < len(sample):
        cur_x = signal[sample[start]:sample[start+interval], :]
        cur_x = (cur_x - mean) / std

        if cur_x.shape[0] >= length:
            cur_x = cur_x[:length, :]
        else:
            cur_x = np.vstack((cur_x, np.zeros(shape=(length-cur_x.shape[0], 2))))

        test_x = np.vstack((test_x, [cur_x]))

        start += interval

    return test_x[1:]


def to_num(array):
    return np.array([np.argmax(each) for each in array])


def smooth(array):
    thresh = array.shape[0]
    for i in range(1, thresh):
        if array[i] != array[i-1]:
            if i+2 < thresh and (array[i+1:i+3] == array[i-1]).all() or i-2 >= 0 and i+1 < thresh and array[i-2] == array[i-1] and array[i+1] == array[i-1]:
                array[i] = array[i-1]
    return array


def model_test(test_x, model, one_hot=True):
    predict_y = model.predict(test_x)

    if one_hot:
        return to_num(predict_y)
    return predict_y


def challenge_entry(sample_path):
    """
    This is a baseline method.
    """

    sig, _, fs = load_data(sample_path)
    # sig = sig[:, 1]
    end_points = []

    r_peaks = qrs_detect(sig[:, 1], fs=200)
    print(r_peaks)

    test_x = load_test_data(sig, r_peaks)
    pred_y = model_test(test_x, model)
    is_af = smooth(pred_y)

    if np.sum(is_af) == len(is_af):
        end_points.append([0, len(sig)-1])
    elif np.sum(is_af) != 0:
        state_diff = np.diff(is_af)
        start_r = np.where(state_diff==1)[0] + 1
        end_r = np.where(state_diff==-1)[0] + 1

        if is_af[0] == 1:
            start_r = np.insert(start_r, 0, 0)
        if is_af[-1] == 1:
            end_r = np.insert(end_r, len(end_r), len(is_af)-1)
        start_r = np.expand_dims(start_r, -1)
        end_r = np.expand_dims(end_r, -1)
        start_end = np.concatenate((r_peaks[start_r], r_peaks[end_r]), axis=-1).tolist()
        end_points.extend(start_end)
        
    pred_dcit = {'predict_endpoints': end_points}
    
    return pred_dcit


if __name__ == '__main__':
    DATA_PATH = sys.argv[1]
    RESULT_PATH = sys.argv[2]
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)

    model = load_model('model/single_model.h5')
    mean = np.load('model/mean.npy')
    std = np.load('model/std.npy')
    length = 1024

    test_set = open(os.path.join(DATA_PATH, 'RECORDS'), 'r').read().splitlines()
    for i, sample in enumerate(test_set):
        print(sample)
        sample_path = os.path.join(DATA_PATH, sample)
        pred_dict = challenge_entry(sample_path)

        save_dict(os.path.join(RESULT_PATH, sample+'.json'), pred_dict)
