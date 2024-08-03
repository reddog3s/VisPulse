import os
import cv2
from det_seg_track.DetSegTrack import DetSegTrack
import time
import numpy as np
import pandas as pd
import argparse
import re
import json
from det_seg_track.utils import bbox_to_xywh, mask_to_bbox, getKeypointsVis, getPossibleFaceArea

#extract int numbers from string
def num_sort(test_string):
    return list(map(int, re.findall(r'\d+', test_string)))[0]

parser = argparse.ArgumentParser(description='Project')
parser.add_argument('--detector', required=True,
                    help='detector')
parser.add_argument('--tracker', required=True,
                    help='tracker')
parser.add_argument('--segmentator', required=False, default=None,
                    help='segmentator')
parser.add_argument('--use_deployed_model',type=bool, required=False, default=False,
                    help='use_deployed_model')
parser.add_argument('--save_json', type=bool, default=False, required=False,
                    help='save results')
parser.add_argument('--save_vis', type=bool, default=False, required=False,
                    help='save results')

if __name__ == "__main__":
    args = parser.parse_args()
    detector = args.detector
    tracker = args.tracker
    segmentator = args.segmentator
    use_deployed_model = args.use_deployed_model
    save_json = args.save_json
    save_vis = args.save_vis
else:
    vid_name = 'IMG_9297'
    #vid_path = os.path.join('/mnt','c','Users','Dell','studia-pliki-robocze','magisterka','src', vid_name + '.mov')
    # python main.py --filepath /mnt/c/Users/Dell/studia-pliki-robocze/magisterka/src/IMG_4827.mp4 --detector rtmo-l --tracker bytetrack --segmentator mobile_sam
    # python main.py --filepath /mnt/d/test/iphone/video/IMG_9297.mp4 --detector rtmo-l --tracker bytetrack --segmentator mobile_sam
    vid_path = os.path.join('/mnt','d', 'test','iphone','video' , vid_name + '.mp4')
    detector = 'rtmo-l'
    tracker = 'bytetrack'
    segmentator = 'mobile_sam'
    use_deployed_model = False
    save_mode = 'vid'
    hr_estimation_method = None

out_dir = './results/keypoints'
#out_img_dir = './results/images'
out_img_dir = os.path.join('/mnt','d', 'test','PoseTrack21-main','preds')
input_path = os.path.join('/mnt','d', 'test','PoseTrack21-main','downloads')
gt_base_path = os.path.join(input_path, 'posetrack_data','val')
img_dirs = os.listdir(gt_base_path)

try:
    for dir_name in img_dirs:
        det_seg_track_module = DetSegTrack(detector, tracker, segmentator, use_deployed_model = use_deployed_model, validate_person= False)
        # read gt data and init results
        gt_path = os.path.join(gt_base_path, dir_name)
        with open(gt_path, 'r') as f:
            gt_data = json.load(f)

        results = {
            'images': gt_data['images'],
            'annotations': [],
            'categories': gt_data['categories']
        }

        # gt annotations to df
        annotations_gt = pd.DataFrame.from_dict(gt_data['annotations'])

        for img_data in gt_data['images']:
            if img_data['is_labeled']:
                img_name = img_data['file_name']
                print('Frame: ', img_name, '\n')

                img_id = img_data['image_id']
                # img_gt_df = annotations_gt[annotations_gt['image_id'] == img_id]

                image_path = os.path.join(input_path, img_name)
                frame = cv2.imread(image_path)
                annotated_frame, person_list = det_seg_track_module.estimate(frame)

                for person in person_list:
                    # gt_person_df = img_gt_df[img_gt_df['track_id'] == person.tracker_id]
                    if person.tracker_id is None:
                        person.tracker_id = 0

                    possible_face_mask = getPossibleFaceArea(person, frame.shape)
                    if possible_face_mask is not None:
                        mask_bbox = mask_to_bbox(possible_face_mask)
                    else:
                        mask_bbox = [0,0,0,0]


                    person_dict = {
                        'bbox': bbox_to_xywh(person.bbox),
                        'bbox_head': mask_bbox,
                        'category_id': 1,
                        'id': int(str(img_id) + '0' + str(person.tracker_id)),
                        'image_id': int(img_id),
                        'keypoints': getKeypointsVis(person.keypoints, frame.shape),
                        'person_id': int(person.tracker_id),
                        'track_id': int(person.tracker_id)
                        }
                    results['annotations'].append(person_dict)
                        
                if save_vis:
                    # bbox = 
                    #annotated_frame = cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                    #                                  (0, 255, 255), 5)

                    file_name = img_name.split("/")[-1]
                    out_path_img = os.path.join(out_img_dir, file_name)
                    cv2.imwrite(out_path_img, annotated_frame)
            else:
                continue

        if save_json:
            out_dir_path = os.path.join(out_dir, dir_name)
            with open(out_dir_path , 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
except KeyboardInterrupt:
    print('\nKeyboard interrupt \n')
    if save_json:
        out_dir_path = os.path.join(out_dir, dir_name)
        with open(out_dir_path , 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)