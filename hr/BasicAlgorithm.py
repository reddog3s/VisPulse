import cv2
import numpy as np
from scipy.signal import butter, filtfilt

from hr.methods.green import green
from hr.methods.pos import POS_WANG
from hr.methods.exg import ExG
from hr.hr_utils import calculate_fft_hr, calculate_peak_hr

def BasicAlgorithm(face, method = 'green', fs = 30, hr_estimation = 'fft'):
    if method == 'green':
        signal = green(face)
    elif method == 'ExG':
        signal = ExG(face)
    elif method == 'POS':
        signal = POS_WANG(face)

    [b, a] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
    signal_filtered = filtfilt(b, a, np.double(signal))

    if hr_estimation == 'fft':
        hr = calculate_fft_hr(signal_filtered, fs = fs)
    elif hr_estimation == 'peaks':
        hr = calculate_peak_hr(signal_filtered, fs = fs)

    return hr
