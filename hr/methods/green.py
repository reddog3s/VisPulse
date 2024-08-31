from hr.hr_utils import process_video, detrend_MCWS

def green(frames):
    #processed_data = process_video(frames)
    #processed_data = detrend_MCWS(frames)
    BVP = frames[:, 1]
    return BVP