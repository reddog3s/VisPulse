import pandas as pd
import os
import cv2
from det_seg_track.utils import mask_to_bbox

vid_csv_path = os.path.join('/mnt','d', 'test', 'iphone','vid_data_temp.csv')
vid_data = pd.read_csv(vid_csv_path, date_format='%Y-%m-%d %H:%M:%S%z',
                             parse_dates=['vid_start','vid_end'],
                             dtype={
                                'vid_id': 'int',
                                'vid_start': 'string',
                                'vid_end': 'string',
                                'person_left': 'int',
                                'watch_left': 'int',
                                'person_right': 'int',
                                'watch_right': 'int',
                                'exercise': 'string',
                                'file_path': 'string'
                            })
vid_size = (1920,1080)

for vid_id in vid_data['vid_id']:
    curr_vid_data = vid_data[vid_data['vid_id'] == vid_id]
    num_person = 0
    if curr_vid_data['watch_right'].iloc[0] != -1:
        num_person += 1
    if curr_vid_data['watch_left'].iloc[0] != -1:
        num_person += 1
    vid_filename = curr_vid_data['file_path'].iloc[0]
    vid_name = vid_filename.split('.')[0]

    base_path = os.path.join('./results', 'images', vid_name)
    files = os.listdir(base_path)
    i = 0
    for file in files:
        if i == 0:
            # temp for 2 person
            if len(bbox) > 0:
                person_id = curr_vid_data['person_left'].iloc[0]
                if num_person > 1:
                    if vid_size[0]/2 < bbox[0] and curr_vid_data['person_right'].iloc[0] > 0:
                        person_id = curr_vid_data['person_right'].iloc[0]
            else:
                person_id = 'nomask'
        path_mask = os.path.join(base_path, file)
        mask = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE)
        bbox = mask_to_bbox(mask)
        
        el = file.split('_')
        id = el[1]

        new_name = 'img_' + id + '_' + str(person_id) + '.jpg'
        print(vid_name,' ',new_name)
        os.rename(path_mask, os.path.join(base_path, new_name))
