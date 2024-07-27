
from ultralytics.models.fastsam import FastSAMPrompt
from ultralytics import FastSAM, SAM
import numpy as np
import os
import cv2

from det_seg_track.utils import getPoints, getEuclideanDistance, validatePoint

class Segmentator:
    def __init__(self, segmentator_name):
        self.segmentator_name = segmentator_name
        if (segmentator_name == 'FastSAM-s'):
            segmentator_path = os.path.join('./checkpoints','FastSAM-s.pt')
            self.seg_model = FastSAM(segmentator_path)
        if (segmentator_name == 'FastSAM-x'):
            segmentator_path = os.path.join('./checkpoints','FastSAM-x.pt')
            self.seg_model = FastSAM(segmentator_path)
        elif (segmentator_name == 'sam_b'):
            segmentator_path = os.path.join('./checkpoints', 'sam_b.pt')
            self.seg_model = SAM(segmentator_path)
        elif (segmentator_name == 'sam_l'):
            segmentator_path = os.path.join('./checkpoints', 'sam_l.pt')
            self.seg_model = SAM(segmentator_path)
        elif (segmentator_name == 'mobile_sam'):
            segmentator_path = os.path.join('./checkpoints', 'mobile_sam.pt')
            self.seg_model = SAM(segmentator_path)
    
    def initFastSAM(self, frame):
        if ('FastSAM' in self.segmentator_name):
            segmentation_results = self.seg_model(frame, device="cpu", retina_masks=True, imgsz=1024,
                                                    conf=0.2, iou=0.9)
            self.prompt_process = FastSAMPrompt(frame, segmentation_results, device="cpu")
            
    def getFaceMask(self, frame, person):
        if ('FastSAM' in self.segmentator_name):
            points, labels = getPoints(person, frame.shape)
            ann = self.prompt_process.point_prompt(points=points, pointlabel=labels)
        else:
            points, labels = getPoints(person, frame.shape)
            ann = self.seg_model(frame, points=points, labels=labels)

        boolean_mask = ann[0].masks.data.numpy()[0]
        mask_area = boolean_mask.sum()
        image_area = boolean_mask.shape[0] * boolean_mask.shape[1]

        # if mask is less than 10% of image area, it's considered invalid
        if ((mask_area/image_area) > 0.1):
            mask = np.float32(np.multiply(boolean_mask, 255))

            # if eyes are visible, use them to create possible face area mask
            # if not, use bbox
            left_eye = person.getBodyPart('LeftEye')
            right_eye = person.getBodyPart('RightEye')
            nose = np.array(person.getBodyPart('Nose'))
            if validatePoint(right_eye, frame.shape) and validatePoint(left_eye, frame.shape):
                mask = checkIfFitsInFaceArea(mask, nose, 
                                             right_eye=np.array(right_eye), 
                                             left_eye=np.array(left_eye))
            elif validatePoint(right_eye, frame.shape):
                mask = checkIfFitsInFaceArea(mask, nose, right_eye=np.array(right_eye))
            elif validatePoint(left_eye, frame.shape):
                mask = checkIfFitsInFaceArea(mask, nose, left_eye=np.array(left_eye))
            else:
                mask = checkIfFitsInBBox(mask, person.bbox)
            # morphological filtration
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(21,21))
            mask = morphologicalFiltration(mask, kernel)
        else:
            mask = None
        return mask

def morphologicalFiltration(mask, kernel):
    """Perform opening and closing
    """
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def checkIfFitsInBBox(mask, bbox):
    """check if mask is in bbox
    """
    bbox = [ int(x) for x in bbox ]
    bbox_mask = np.zeros(mask.shape, np.float32)
    bbox_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 255

    mask_cropped_to_bbox = cv2.bitwise_and(mask, bbox_mask, mask = None)

    return mask_cropped_to_bbox

def checkIfFitsInFaceArea(mask, nose, left_eye = None, right_eye = None):
    """Check if mask fits in circle, which \n
    center is nose \n
    radius is doubled distance between eyes and nose
    """
    radius_list = []
    if left_eye is not None:
        radius_list.append(getEuclideanDistance(nose, left_eye))
    if right_eye is not None:
        radius_list.append(getEuclideanDistance(nose, right_eye))

    radius_mean = int(np.mean(radius_list))
    radius = radius_mean * 3

    face_mask = np.zeros(mask.shape, np.float32)
    face_mask = cv2.circle(face_mask, nose, radius, 255, -1)
    mask_cropped_to_bbox = cv2.bitwise_and(mask, face_mask, mask = None)

    return mask_cropped_to_bbox


