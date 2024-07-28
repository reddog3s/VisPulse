import numpy as np
from scipy import sparse
from scipy.signal import convolve2d, find_peaks, periodogram

def detrend(input_signal, lambda_value):
    """Detrend PPG signal."""
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = sparse.spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    detrended_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return detrended_signal

def detrend_MCWS(input_signal, window = 1, fps = 30):
    """Input signal has shape [n, 3]
    """
    print(input_signal.shape)
    w = round(window * fps);    
    n = convolve2d(np.ones((input_signal.shape[1], input_signal.shape[0])), np.ones((1, w)), 'same')
    meanIntensity = convolve2d(np.transpose(input_signal), np.ones((1, w)), 'same') / n
    sig1 = (np.transpose(input_signal) - meanIntensity) / meanIntensity
    return np.transpose(sig1)

def process_video(frames, format = None):
    "Calculates the average value of each frame."
    RGB = []
    for frame in frames:
        summation = np.sum(np.sum(frame, axis=0), axis=0)
        RGB.append(summation / (frame.shape[0] * frame.shape[1]))
    RGB = np.asarray(RGB)

    if format == '3d':
        RGB = RGB.transpose(1, 0).reshape(1, 3, -1)
    
    return np.asarray(RGB)

def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def calculate_peak_hr(ppg_signal, fs = 30):
    """Calculate heart rate based on PPG using peak detection."""
    ppg_peaks, _ = find_peaks(ppg_signal)
    hr_peak = 60 / (np.mean(np.diff(ppg_peaks)) / fs)
    return hr_peak

def calculate_fft_hr(ppg_signal, fs=30, low_bpm=45, high_bpm=180):
    # calculate fft
    ppg_signal = np.expand_dims(ppg_signal, 0)
    N = _next_power_of_2(ppg_signal.shape[1])
    print('N: ', N)
    f_ppg, pxx_ppg = periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)

    # filter results between lowest and highest possible bpm
    low_pass = low_bpm / 60
    high_pass = high_bpm / 60
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60

    return fft_hr