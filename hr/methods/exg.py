import numpy as np
from hr.hr_utils import process_video, detrend_MCWS

def ExG(frames):
    # processed_data = process_video(frames)
    # processed_data = detrend_MCWS(processed_data)
    R_intensity = processed_data[:, 2]
    G_intensity = processed_data[:, 1]
    B_intensity = processed_data[:, 0]

    divider = B_intensity + G_intensity + R_intensity
    B_normalized = B_intensity / divider
    G_normalized = G_intensity / divider
    R_normalized = R_intensity / divider

    return 2 * G_normalized - R_normalized - B_normalized