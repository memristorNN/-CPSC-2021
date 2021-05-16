#!/usr/bin/env python3

import numpy as np
import os
import sys
os.chdir(sys.path[0])

import wfdb
from utils import qrs_detect, comp_cosEn, save_dict
from tensorflow.keras.models import load_model
import tensorflow as tf
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


def load_test_data(sample, length, fs):
    sample = np.diff(sample) / fs

    t_x = np.reshape(sample[:sample.shape[0] // length * length], (-1, length))
    s_x = sample[-length:]
    interval = sample.shape[0] - sample.shape[0] // length * length

    return t_x, s_x, interval


def dice_coef(y_true, y_pred, smooth=1):
    intersection = tf.keras.backend.sum(y_true * y_pred, axis=-1)
    union = tf.keras.backend.sum(y_true, axis=-1) + tf.keras.backend.sum(y_pred, axis=-1)
    return tf.keras.backend.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred, smooth=1)


def calc_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)


def to_num(array):
    return np.array([np.argmax(each) for each in array])


def smooth(array):
    thresh = array.shape[0]
    for i in range(1, thresh):
        if array[i] != array[i-1]:
            if i+2 < thresh and (array[i+1:i+3] == array[i-1]).all() or i-2 >= 0 and i+1 < thresh and array[i-2] == array[i-1] and array[i+1] == array[i-1]:
                array[i] = array[i-1]
    return array


def model_test(test_x, model, interval):
    predict_y = model.predict(np.expand_dims(test_x, 2))
    predict_y[predict_y >= 0.5] = 1
    predict_y[predict_y < 0.5] = 0

    s_y = predict_y[-1]
    predict_y = predict_y[:-1].ravel()
    if interval > 0:
        predict_y = np.hstack((predict_y, s_y[-interval:]))

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

    cur_x, single_x, interval = load_test_data(r_peaks, length, fs)

    pred_y = model_test(np.vstack((cur_x, single_x)), model, interval)
    # is_af = smooth(pred_y)
    is_af = pred_y

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

    model = load_model('model/single_model.h5', custom_objects={'calc_loss': calc_loss})
    length = 12

    test_set = open(os.path.join(DATA_PATH, 'RECORDS'), 'r').read().splitlines()
    for i, sample in enumerate(test_set):
        print(sample)
        sample_path = os.path.join(DATA_PATH, sample)
        pred_dict = challenge_entry(sample_path)

        save_dict(os.path.join(RESULT_PATH, sample+'.json'), pred_dict)
