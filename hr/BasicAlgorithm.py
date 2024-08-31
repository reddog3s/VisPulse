import cv2
import numpy as np
from scipy.signal import butter, filtfilt

from hr.methods.green import green
from hr.methods.pos import POS_WANG
from hr.methods.exg import ExG
from hr.methods.ica import ICA_POH
from hr.methods.chrom import CHROME_DEHAAN
from hr.hr_utils import calculate_fft_hr, calculate_peak_hr, detrend_MCWS

def BasicAlgorithm(rgb, method = 'green', fs = 30, hr_estimation = 'fft'):
 
    rgb_detrended = detrend_MCWS(rgb, fps = fs)
    if method == 'green':
        signal = green(rgb_detrended)
    elif method == 'ExG':
        signal = ExG(rgb_detrended)
    elif method == 'POS':
        signal = POS_WANG(rgb_detrended, fs)
    elif method == 'ICA':
        signal = ICA_POH(rgb_detrended, fs)
    elif method == 'CHROM':
        signal = CHROME_DEHAAN(rgb_detrended, fs)

    [b, a] = butter(1, [0.75 / fs * 2, 3 / fs * 2], btype='bandpass')
    signal_filtered = filtfilt(b, a, np.double(signal))

    if hr_estimation == 'fft':
        hr, mask_ppg, mask_pxx  = calculate_fft_hr(signal_filtered, fs = fs)
    elif hr_estimation == 'peaks':
        hr = calculate_peak_hr(signal_filtered, fs = fs)

    return hr, rgb, rgb_detrended, signal, signal_filtered, mask_ppg, mask_pxx
