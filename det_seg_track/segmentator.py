
from ultralytics.models.fastsam import FastSAMPrompt
from ultralytics import FastSAM, SAM
import numpy as np
import os
import cv2

from det_seg_track.utils import getPoints, getPossibleFaceArea

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
    
    def extractSkinByHSV(self, frame):
        img_BGRA = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        B = img_BGRA[:,:,0]
        G = img_BGRA[:,:,1]
        R = img_BGRA[:,:,2]
        mask1 = (R > G) * (R > B) * (np.abs(R - G) > 15)
        mask1 = np.float32(np.multiply(mask1, 255))
        low_thresh = (20, 40, 95, 15)
        high_thresh = (255,255,255, 255)
        BGRA_mask = cv2.inRange(img_BGRA, low_thresh, high_thresh) 
        mask_after_BGRA = cv2.bitwise_and(BGRA_mask, mask1, mask = None)

        img_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        low_thresh = (0, 20, 70)
        high_thresh = (20,255,255)
        HSV_mask = cv2.inRange(img_HSV, low_thresh, high_thresh)
        mask_end = cv2.bitwise_and(mask_after_BGRA, HSV_mask, mask = None)
        return mask_end

    def initFastSAM(self, frame):
        if ('FastSAM' in self.segmentator_name):
            segmentation_results = self.seg_model(frame, device="cpu", retina_masks=True, imgsz=1024,
                                                    conf=0.2, iou=0.9)
            self.prompt_process = FastSAMPrompt(frame, segmentation_results, device="cpu")
            
    def getFaceMask(self, frame, person):
        if ('FastSAM' in self.segmentator_name):
            points, labels = getPoints(person, frame.shape)
            ann = self.prompt_process.point_prompt(points=points, pointlabel=labels)
            boolean_mask = ann[0].masks.data.numpy()[0]
        elif ('hsv' == self.segmentator_name):
            boolean_mask = self.extractSkinByHSV(frame)
        else:
            points, labels = getPoints(person, frame.shape)
            ann = self.seg_model(frame, points=points, labels=labels)
            boolean_mask = ann[0].masks.data.numpy()[0]
        # mask_area = boolean_mask.sum()
        # image_area = boolean_mask.shape[0] * boolean_mask.shape[1]

        # if mask is less than 10% of image area, it's considered invalid
        
        mask = np.float32(np.multiply(boolean_mask, 255))

        # if eyes or ears are visible, use them to create possible face area mask
        # if not, use bbox
        possible_face_mask = getPossibleFaceArea(person, mask.shape)
        if possible_face_mask is not None:
            mask = cropMask(mask, possible_face_mask)
        else:
            mask = cropMask(mask, getBBoxMask(person.bbox, mask.shape))
    
        # morphological filtration
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(21,21))
        mask = morphologicalFiltration(mask, kernel)

        if np.sum(mask) <= 0:
            if possible_face_mask != None and np.sum(possible_face_mask) > 0:
                mask = possible_face_mask


        return mask

def morphologicalFiltration(mask, kernel):
    """Perform opening and closing
    """
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def getBBoxMask(bbox, frame_shape):
    """check if mask is in bbox
    """
    bbox = [ int(x) for x in bbox ]
    bbox_mask = np.zeros(frame_shape, np.float32)
    bbox_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 255

    return bbox_mask

def cropMask(mask, face_mask):
    mask_cropped_to_bbox = cv2.bitwise_and(mask, face_mask, mask = None)
    return mask_cropped_to_bbox



