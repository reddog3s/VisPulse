import os
import cv2
from det_seg_track.utils import ImageAnnotator, Person, mask_to_bbox
from hr.HeartRateEstimator import HeartRateEstimator
from hr.hr_utils import pad_dict_list
import numpy as np
import pandas as pd
import argparse

vid_csv_path = os.path.join('/mnt','d', 'test', 'iphone','vid_data.csv')
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

parser = argparse.ArgumentParser(description='Project')
parser.add_argument('--filepath', required=True,
                    help='path to videos')
parser.add_argument('--hr_method', required=True, 
                    help='hr estimation method')
parser.add_argument('--use_deployed_model',type=bool, required=False, default=False,
                    help='use_deployed_model')
parser.add_argument('--save_mode', required=False, default=None,
                    help='use_deployed_model')

if __name__ == "__main__":
    args = parser.parse_args()
    vid_path = args.filepath
    save_mode = args.save_mode
    hr_estimation_method = args.hr_method

out_path = os.path.join('./results')
out_path_hr = os.path.join(out_path, 'hr_results', 'predicted')
annotator = ImageAnnotator()

if save_mode is not None:
    save_vis = True
else:
    save_vis = False

for vid_id in vid_data['vid_id']:
    person_ids = []
    curr_vid_data = vid_data[vid_data['vid_id'] == vid_id]
    num_person = 0
    if curr_vid_data['watch_right'].iloc[0] != -1:
        num_person += 1
        person_ids.append(curr_vid_data['person_right'].iloc[0])
    if curr_vid_data['watch_left'].iloc[0] != -1:
        num_person += 1
        person_ids.append(curr_vid_data['person_left'].iloc[0])
    vid_filename = curr_vid_data['file_path'].iloc[0]
    curr_vid_path = os.path.join(vid_path, vid_filename) 
    #print(curr_vid_path)

    cap = cv2.VideoCapture(curr_vid_path)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    print('video fps: ',fps)
    #vid_size = (1080,1920)
    vid_size = (1920,1080)
    vid_name = vid_filename.split('.')[0]
    out_path_vid  = os.path.join('/mnt','d','test','iphone', 'videos', vid_name + '_out.avi')
    out = cv2.VideoWriter(out_path_vid, cv2.VideoWriter_fourcc('M','J','P','G'), fps, vid_size)
    hr_module = HeartRateEstimator(hr_estimation_method, fps = fps)
    times = []
    hr = []
    hr_data = {
                'frame_id': [],
                'time': [],
                'person_id': [],
                'person_bbox': [],
                'hr': []
            }
    # Loop through the video frames

    i = 0
    try: 
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                print('Frame ', i, '\n')
                person_list = []
                for person_id in person_ids:
                    person = Person()
                    path_mask = os.path.join(out_path, 'images',vid_name, "img_" + str(i) + "_" + str(person_id) + ".jpg")
                    person.mask = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE)
                    person.bbox = mask_to_bbox(person.mask)
                    if len(person.bbox) == 0:
                        person.bbox = [0, 0, 0, 0]
                    person.tracker_id = person_id
                    person_list.append(person)
                

                hr_results = hr_module.estimate(frame, i, person_list)

                for person_hr in hr_results:
                    hr_data['frame_id'].append(i)
                    hr_data['time'].append(i/fps)
                    hr_data['person_id'].append(person_hr['person_id'])
                    hr_data['hr'].append(person_hr['hr'])
                    for person in person_list:
                        if person.tracker_id == person_hr['person_id']:
                            person.hr = person_hr['hr']
                            person.image_id = i
                            if len(person.bbox) > 0:
                                bbox_str = str(person.bbox[0]) + ' ' + str(person.bbox[1]) + ' ' + str(person.bbox[2]) + ' ' + str(person.bbox[3])
                            else:
                                bbox_str = '0 0 0 0'
                            hr_data['person_bbox'].append(bbox_str)
                    
                if save_vis:
                    annotator.initAnnotator(frame, convertRGBToBGR = False)
                    for person in person_list:
                        annotator.annotateImage(person, showTracker = True, showNose = False, showBBox = True, showMask = True, showShoulders = False)
                    out.write(annotator.annotated_frame)
                i+=1
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Break the loop if the end of the video is reached
                break
    except KeyboardInterrupt:
        print('\nKeyboard interrupt \n')
        
    hr_data = pad_dict_list(hr_data, -1)
    print('Mean time: ', np.mean(times))
    df = pd.DataFrame.from_dict(hr_data)
    person = df[df['person_id'] == 1]
    print('Mean hr for person 1: ', np.mean(person['hr']))
    out_path_file = os.path.join(out_path_hr, 'hr_' + str(vid_id) + '_'+ hr_estimation_method + '_pred.csv')
    df.to_csv(out_path_file,index = False)
    # Release the video capture object and close the display window
    cap.release()
    out.release()
    del hr_module
del annotator