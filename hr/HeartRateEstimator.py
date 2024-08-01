import numpy as np
from hr.BasicAlgorithm import BasicAlgorithm
from hr.hr_utils import process_video
import pandas as pd
import collections

class HeartRateEstimator:
    def __init__(self, method = 'POS', max_buffer_len = 512, fps = 30):
        self.max_buffer_len = max_buffer_len
        self.circular_buffer = collections.deque(maxlen=self.max_buffer_len)
        self.method = method
        self.fps = fps

    def estimate(self, frame, frame_id, person_list):
        rgb_data_person_list = []
        #print(len(self.circular_buffer))
        for person in person_list:
            # to 0 1 mask from 0 255
            mask = person.mask
            if mask is not None:
                mask = np.divide(mask, 255)
                # to bgr
                colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
                colored_mask = np.moveaxis(colored_mask, 0, -1)
                # extract colored face from mask
                face = frame * colored_mask
                # append tracker_id and rgb_data to buffer
                if self.method == 'POS' or self.method == 'ICA':
                    rgb_data_person_list.append((person.tracker_id, process_video([face], format = '3d')))
                else:
                    rgb_data_person_list.append((person.tracker_id, process_video([face])))
            else:
                rgb_data_person_list.append((person.tracker_id, np.array([0, 0, 0], ndmin=2)))

        frame_data = {
            'frame_id': frame_id,
            'person_list': rgb_data_person_list
        }
        self.circular_buffer.append(frame_data)

        hr_data = []
        
        if len(self.circular_buffer) >= self.max_buffer_len:
            data = {
                'frame_id': [],
                'person_id': [],
                'rgb_data': []
            }
            for frame_data in self.circular_buffer:
                for person in frame_data['person_list']:
                    data['frame_id'].append(frame_data['frame_id'])
                    data['person_id'].append(person[0])
                    data['rgb_data'].append(person[1])
                
            df = pd.DataFrame.from_dict(data)

            person_ids = df['person_id'].unique()
            for id in person_ids:
                person_df = df[df['person_id'] == id]

                if person_df.shape[0] >= self.max_buffer_len:
                    rgb_data = person_df['rgb_data'].to_list()
                    rgb_data = np.vstack(tuple(rgb_data))
                    print('rgb shape ', rgb_data.shape)
                    hr = BasicAlgorithm(rgb_data, method = self.method, fs = self.fps)
                else:
                    hr = 0
                print('person id ', id)
                print('hr ', hr)
                hr_data.append({
                    'person_id': id,
                    'hr': hr
                })
                
        
        return hr_data