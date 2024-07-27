import os
import cv2
from det_seg_track.DetSegTrack import DetSegTrack
from hr.HeartRateEstimator import HeartRateEstimator
import time
import numpy as np

vid_name = 'IMG_4827'
vid_path = os.path.join('/mnt','c','Users','Dell','studia-pliki-robocze','magisterka','src', vid_name + '.mp4')
out_path = os.path.join('./results')
out_path_vid = os.path.join(out_path, 'videos', vid_name +'_out.avi')

cap = cv2.VideoCapture(vid_path)
vid_size = (1080,1920)
out = cv2.VideoWriter(out_path_vid, cv2.VideoWriter_fourcc('M','J','P','G'), 10, vid_size)
# yolov8s-pose
det_seg_track_module = DetSegTrack('rtmo-l', 'bytetrack', 'sam_l', use_deployed_model = False)
hr_module = HeartRateEstimator()
times = []
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
            hr_results = hr_module.estimate(frame, person_list)
            end = time.time()
            print('Operation time: ', end - start)
            times.append(end - start)
            #for person in person_list:
            #    print(vars(person))
            out_path_img = os.path.join(out_path, 'images', "img_" + str(i) + ".jpg")
            cv2.imwrite(out_path_img, annotated_frame) 
            #out.write(annotated_frame)
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
# Release the video capture object and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()