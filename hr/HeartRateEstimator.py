import numpy as np
from hr.BasicAlgorithm import BasicAlgorithm

class HeartRateEstimator:
    def __init__(self):
        self.frame_buffer = []
        self.person_list_buffer = []
        self.max_buffer_len = 20
    def estimate(self, frame, person_list):
        self.frame_buffer.append(frame)
        self.person_list_buffer.append(person_list)

        hr = 0
        if len(self.frame_buffer) > 20:
            # to 0 1 mask from 0 255
            mask = person_list[0].mask
            mask = np.divide(mask, 255)
            colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
            colored_mask = np.moveaxis(colored_mask, 0, -1)
            # extract colored face from mask
            face = frame * colored_mask

            hr = BasicAlgorithm(face)
        
        return hr