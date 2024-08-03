import os
import cv2
from det_seg_track.DetSegTrack import DetSegTrack
from hr.HeartRateEstimator import HeartRateEstimator
import time
import numpy as np
import pandas as pd
import argparse

vid_size = (1920,1080)
fps = 30
vid_csv_path = os.path.join('/mnt','d', 'test', 'iphone','vid_data.csv')
vid_data = pd.read_csv(vid_csv_path, date_format='%Y-%m-%d %H:%M:%S%z',
                             parse_dates=['vid_start','vid_end'],
                             dtype={
                                'vid_id': 'int',
                                'vid_start': 'string',
                                'vid_end': 'string',
                                'person_left': 'string',
                                'watch_left': 'int',
                                'person_right': 'string',
                                'watch_right': 'int',
                                'exercise': 'string',
                                'file_path': 'string'
                            })

parser = argparse.ArgumentParser(description='Project')
parser.add_argument('--filepath', required=True,
                    help='path to videos')
parser.add_argument('--detector', required=True,
                    help='detector')
parser.add_argument('--tracker', required=True,
                    help='tracker')
parser.add_argument('--segmentator', required=True,
                    help='segmentator')
parser.add_argument('--hr_method', required=True, 
                    help='hr estimation method')
parser.add_argument('--use_deployed_model',type=bool, required=False, default=False,
                    help='use_deployed_model')
parser.add_argument('--save_mode', required=False, default='img',
                    help='use_deployed_model')

if __name__ == "__main__":
    args = parser.parse_args()
    vid_path = args.filepath
    detector = args.detector
    tracker = args.tracker
    segmenetator = args.segmentator
    use_deployed_model = args.use_deployed_model
    save_mode = args.save_mode
    hr_estimation_method = args.hr_method
else:
    vid_name = 'IMG_9297'
    #vid_path = os.path.join('/mnt','c','Users','Dell','studia-pliki-robocze','magisterka','src', vid_name + '.mov')
    # python main.py --filepath /mnt/c/Users/Dell/studia-pliki-robocze/magisterka/src/IMG_4827.mp4 --detector rtmo-l --tracker bytetrack --segmentator mobile_sam
    # python main.py --filepath /mnt/d/test/iphone/video/IMG_9297.mp4 --detector rtmo-l --tracker bytetrack --segmentator mobile_sam
    vid_path = os.path.join('/mnt','d', 'test','iphone','video' , vid_name + '.mp4')
    detector = 'rtmo-l'
    tracker = 'bytetrack'
    segmenetator = 'mobile_sam'
    use_deployed_model = False
    save_mode = 'vid'
    hr_estimation_method = 'POS'

out_path = os.path.join('./results')
out_path_vid  = os.path.join(out_path, 'videos', 'vid_out.avi')
out_path_hr = os.path.join(out_path, 'hr_results', 'predicted')

for vid_id in vid_data['vid_id']:
    curr_vid_data = vid_data[vid_data['vid_id'] == vid_id]
    curr_vid_path = os.path.join(vid_path, curr_vid_data['file_path'].iloc[0])
    print(curr_vid_path)

    cap = cv2.VideoCapture(curr_vid_path)
    #vid_size = (1080,1920)
    out = cv2.VideoWriter(out_path_vid, cv2.VideoWriter_fourcc('M','J','P','G'), 10, vid_size)
    # yolov8s-pose
    det_seg_track_module = DetSegTrack(detector, tracker, segmenetator, use_deployed_model = use_deployed_model)
    hr_module = HeartRateEstimator(hr_estimation_method, fps = fps)
    times = []
    hr = []
    hr_data = {
                'frame_id': [],
                'time': [],
                'person_id': [],
                'hr': []
            }
    # Loop through the video frames

    i = 0
    try: 
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                print('Frame ', i, '\n')
                start = time.time()
                annotated_frame, person_list = det_seg_track_module.estimate(frame)
                hr_results = hr_module.estimate(frame, i, person_list)
                for person_hr in hr_results:
                    hr_data['frame_id'].append(i)
                    hr_data['time'].append(i/fps)
                    hr_data['person_id'].append(person_hr['person_id'])
                    hr_data['hr'].append(person_hr['hr'])
                    
                end = time.time()
                print('Operation time: ', end - start)
                times.append(end - start)
                #for person in person_list:
                #    print(vars(person))
                if save_mode == 'img':
                    out_path_img = os.path.join(out_path, 'images', "img_" + str(i) + ".jpg")
                    cv2.imwrite(out_path_img, annotated_frame)
                elif save_mode == 'vid':
                    out.write(annotated_frame)
                i+=1
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Break the loop if the end of the video is reached
                break
    except KeyboardInterrupt:
        print('\nKeyboard interrupt \n')
        

    print('Mean time: ', np.mean(times))
    df = pd.DataFrame.from_dict(hr_data)
    person = df[df['person_id'] == 1]
    print('Mean hr for person 1: ', np.mean(person['hr']))
    out_path_file = os.path.join(out_path_hr, 'hr_' + str(vid_id) + '_pred.csv')
    df.to_csv(out_path_file,index = False)
    # Release the video capture object and close the display window
    cap.release()
    out.release()