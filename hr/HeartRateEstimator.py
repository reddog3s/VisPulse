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
        self.i = 0

    def estimate(self, frame, frame_id, person_list):
        hr_data = []
        if self.method is not None:
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
                    rgb_data_person_list.append((person.tracker_id, process_video([face])))

            frame_data = {
                'frame_id': frame_id,
                'person_list': rgb_data_person_list
            }
            self.circular_buffer.append(frame_data)
            
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

                    # filter where frames are not consecutive
                    diffs = person_df['frame_id'].diff().fillna(0) > 1
                    invalid_frames = person_df[diffs]
                    # select start_index as first valid frame
                    start_index = invalid_frames['frame_id'].max() + 1
                    if np.isnan(start_index):
                        start_index = 0
                    person_df = person_df[person_df['frame_id'] > start_index]


                    if person_df.shape[0] >= self.max_buffer_len:
                        rgb_data = person_df['rgb_data'].to_list()
                        rgb_data = np.vstack(tuple(rgb_data))
                        #print('rgb shape ', rgb_data.shape)
                        hr, rgb, rgb_detrended, signal, signal_filtered, mask_ppg, mask_pxx = BasicAlgorithm(rgb_data, method = self.method, fs = self.fps)
                        if self.i == 0:
                            self.i+=1
                            # print(rgb_data.shape, rgb_detrended.shape, signal.shape, signal_filtered.shape, mask_ppg.shape, mask_pxx.shape)
                            # data_fft = {
                            #     "freq": mask_ppg[:,0],
                            #     "sig": mask_pxx[:,0]
                            # }
                            # df_fft = pd.DataFrame(data_fft)
                            # df_fft.to_csv('./results/fft.csv')
                            # data_rgb = {
                            #     "rgb": rgb_data[:,1],
                            #     "rgb_detrended": rgb_detrended[:,1],
                            #     "signal": signal,
                            #     "signal_filtered": signal_filtered
                            # }
                            # df_rgb = pd.DataFrame(data_rgb)
                            # df_rgb.to_csv('./results/rgb.csv')
                    else:
                        hr = 0
                    
                    #print('person id ', id)
                    #print('hr ', hr)
                    print('hr ', hr, 'for frames ', person_df['frame_id'].min(),' ', person_df['frame_id'].max())
                    hr_data.append({
                        'person_id': id,
                        'hr': hr
                    })
                
        
        return hr_data