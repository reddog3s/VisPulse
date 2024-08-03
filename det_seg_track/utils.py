import numpy as np
import cv2
from skimage.measure import label, regionprops, find_contours

BODY_PART_MAPPING = {
    "COCO" : {
        "Nose" : 0,
        "RightShoulder" : 6,
        "LeftShoulder" : 5,
        "LeftEye" : 1,
        "RightEye" : 2,
        "RightEar": 4,
        "LeftEar": 3
    }
}

class Person:
    """Person with properties:
            keypoints,
            bbox,
            track_id,
    """
    def __init__(self, keypoints = None, bbox = None, tracker_id = None, keypoint_format = 'COCO'):
        self.keypoints = keypoints
        self.bbox = bbox
        self.tracker_id = tracker_id
        self.keypoint_format = keypoint_format
        self.is_valid = False
        self.mask = None
    def __repr__(self):
        return "person " + str(self.tracker_id)
    def __str__(self):
        return "person " + str(self.tracker_id)

    def fromMMPoseResult(self, instance):
        self.keypoints = instance['keypoints']
        bbox = instance['bbox'][0]
        bbox = np.array(bbox)
        bbox_conf = np.append(bbox, instance['bbox_score'])
        self.bbox = bbox_conf

    def fromMMDeployResult(self, keypoints, bbox, bbox_conf):
        self.keypoints = keypoints
        bbox = np.array(bbox)
        bbox_conf = np.append(bbox, bbox_conf)
        self.bbox = bbox_conf

    def fromUltralyticsResult(self, keypoints, bbox, bbox_conf, tracker):
        self.keypoints = keypoints
        bbox = np.array(bbox)
        bbox_conf = np.append(bbox, bbox_conf)
        self.bbox = bbox_conf
        self.tracker_id = int(tracker)
        
    def getBodyPart(self, body_part_name, reverse=False):
        body_part = None
        body_part_idx = BODY_PART_MAPPING[self.keypoint_format][body_part_name]
        body_part = self.keypoints[body_part_idx]
        body_part = [ int(x) for x in body_part ]

        if reverse:
            body_part.reverse()
        return body_part

    def validatePerson(self, frame_shape, keypoints_conf = None, conf = 0.8):
        is_valid_person = False
        nose = self.getBodyPart('Nose')

        # bbox dim validation
        for i in range(4):
            if self.bbox[i] < 0:
                self.bbox[i] = 0

        bbox_area = (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
        image_area = (frame_shape[0] * frame_shape[1])

        # if bbox is smaller than 20% of image, it's not valid
        # is_bbox_valid = (bbox_area/image_area) > 0.2 and self.bbox[4] > conf
        is_bbox_valid = self.bbox[4] > conf

        # if keypoint conf is less than thresh, make it -1
        if keypoints_conf is not None:
            keypoints_filtered = np.empty((0,2))
            for i, score in enumerate(keypoints_conf):
                keypoint = self.keypoints[i]
                if validatePoint(keypoint, frame_shape, keypoint_conf = score):
                    keypoints_filtered = np.append(keypoints_filtered, np.reshape(self.keypoints[i], (1, 2)), axis = 0)
                else:
                    keypoints_filtered = np.append(keypoints_filtered, [[-1, -1]], axis = 0)
            self.keypoints = keypoints_filtered

        # if nose is not visible, it's not valid
        # print('nose valid: ', validatePoint(nose, frame_shape))
        # print('nose shape: ', nose)
        # print('frame shape: ', frame_shape)
        if (validatePoint(nose, frame_shape) and is_bbox_valid):
            self.is_valid = True
            is_valid_person = True

        return is_valid_person

class ImageAnnotator:
    def __init__(self, frame, annotated_frame = None, convertRGBToBGR = False):
        self.colors = [(0,255,0),
                       (0,0,255),
                       (255,0,0),
                       (125,255,0),
                       (50, 168, 153)
                       ]
        self.frame = frame
        if (annotated_frame is not None):
            # consider that frame is not in bgr format
            if convertRGBToBGR:
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            self.annotated_frame = annotated_frame
    
    def annotateImage(self, person, showTracker = True, showNose = True, showBBox = True, showMask = True, showShoulders = True):
        if showBBox:
            bbox = [ int(x) for x in person.bbox ]
            self.annotated_frame = cv2.circle(self.annotated_frame, (bbox[0], bbox[1]), 
                                                10, (0, 0, 255), -1)
            self.annotated_frame = cv2.circle(self.annotated_frame, (bbox[2], bbox[3]), 
                                                10, (0, 0, 255), -1)
            self.annotated_frame = cv2.rectangle(self.annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                                                 (0, 0, 255), 5)
        if showNose:
            self.annotated_frame = cv2.circle(self.annotated_frame, tuple(person.getBodyPart('Nose')), 
                                                10, (0, 0, 255), -1)
        if showShoulders:
            self.annotated_frame = cv2.circle(self.annotated_frame, tuple(person.getBodyPart('LeftShoulder')), 
                                                10, (0, 255, 255), -1)
            self.annotated_frame = cv2.circle(self.annotated_frame, tuple(person.getBodyPart('RightShoulder')), 
                                                10, (0, 255, 255), -1)
        if showTracker:
            self.annotated_frame = cv2.putText(self.annotated_frame, str(person.tracker_id),
                                                tuple(person.getBodyPart('Nose')), thickness=5, color=(0, 0, 0), 
                                                fontScale=5.0, fontFace=cv2.FONT_HERSHEY_SIMPLEX)
        if showMask and person.mask is not None:
            color = (0,0,255)
            if (person.tracker_id is not None):
                if (person.tracker_id < len(self.colors)):
                    color = self.colors[person.tracker_id]
            self.annotated_frame = overlay(self.annotated_frame, person.mask, color=color, alpha=0.3)
        return self.annotated_frame

def overlay(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    #color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if (resize is not None):
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined

def validatePoint(point, frame_shape, keypoint_conf = None, conf_threshold = 0.1):
    is_valid = False
    is_valid_shape = False
    if (point[0] <= frame_shape[1] and point[1] <= frame_shape[0]):
        if (point[0] > 0 and point[1] > 0):
            is_valid_shape = True
    
    is_valid_conf = False
    if keypoint_conf is not None:
        if keypoint_conf > conf_threshold:
            is_valid_conf = True
    else:
        is_valid_conf = True

    if (is_valid_conf and is_valid_shape):
        is_valid = True
    return is_valid

def getPoints(person, frame_shape, include_negative = True, reverse = False):
    points = []
    labels = []
    negative_points = ['RightShoulder', 'LeftShoulder']
    points.append(person.getBodyPart('Nose', reverse = reverse))
    labels.append(1)

    if include_negative:
        for part in negative_points:
            point = person.getBodyPart(part, reverse = reverse)
            if validatePoint(point, frame_shape):
                points.append(point)
                labels.append(0)

    return points, labels

def getEuclideanDistance(point1, point2):
    return np.linalg.norm(point1-point2)

def bbox_ious(boxes1, boxes2):
    """
    Compute intersection over union (iou) for predicted and gt bounding boxes
    Arguments:
    :param boxes1: N1 X 4, [xmin, ymin, xmax, ymax]
    :param boxes2: N2 X 4, [xmin, ymin, xmax, ymax]

    Return:
    iou - list, N1 X N2
    """
    #if boxes1 is empty, return array of -1, length of boxes2
    if len(boxes1) == 0:
        return np.ones([len(boxes2), 1]) * (-1)

    #if boxes2 is empty, return array of -1, length of boxes1
    if len(boxes2) == 0:
        return np.ones([len(boxes1), 1]) * (-1)
    b1x1, b1y1 = np.split(boxes1[:, :2], 2, axis=1) #min x min y from bbox1
    b1x2, b1y2 = np.split(boxes1[:, 2:4], 2, axis=1) #max x max y from bbox1
    b2x1, b2y1 = np.split(boxes2[:, :2], 2, axis=1) #min x min y from bbox2
    b2x2, b2y2 = np.split(boxes2[:, 2:4], 2, axis=1) #max x max y from bbox2

    dx = np.maximum(np.minimum(b1x2, np.transpose(b2x2)) - np.maximum(b1x1, np.transpose(b2x1)), 0) #max (min from max(1,2) - max from min(1,2)) x
    dy = np.maximum(np.minimum(b1y2, np.transpose(b2y2)) - np.maximum(b1y1, np.transpose(b2y1)), 0) #max (min from max(1,2) - max from min(1,2)) y
    intersections = dx * dy # compute area of rectangle (intersections)

    areas1 = (b1x2 - b1x1) * (b1y2 - b1y1) #max x1 - min x1 * max y1 - min y1, compute area of rectangle (bbox)
    areas2 = (b2x2 - b2x1) * (b2y2 - b2y1)
    unions = (areas1 + np.transpose(areas2)) - intersections #unions is equal to sum of areas minus intersection

    return intersections / unions

def mask_ious(mask1, mask2):
    """Calculate iou between 2 binary masks
    """
    intersection = cv2.bitwise_and(mask1, mask2, mask = None)
    union = cv2.bitwise_or(mask1, mask2, mask = None)

    return np.sum(intersection) / np.sum(union)

def bbox_to_xywh(bbox):
    """From xyxy to xywh
    """
    bbox_xywh = []
    bbox_xywh.append(bbox[0])
    bbox_xywh.append(bbox[1])
    bbox_xywh.append(abs(bbox[0] - bbox[2]))
    bbox_xywh.append(abs(bbox[1] - bbox[3]))
    return bbox_xywh

def bbox_to_xyxy(bbox):
    """From xywh to xyxy
    """
    bbox_xyxy = []
    bbox_xyxy.append(bbox[0])
    bbox_xyxy.append(bbox[1])
    bbox_xyxy.append(bbox[0] + bbox[2])
    bbox_xyxy.append(bbox[1] + bbox[3])
    return bbox_xyxy

""" Convert a mask to border image """
def mask_to_border(mask):
    h, w = mask.shape
    border = np.zeros((h, w))

    contours = find_contours(mask, 128)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255

    return border

""" Mask to bounding boxes """
def mask_to_bbox(mask):
    bboxes = []

    mask = mask_to_border(mask)
    lbl = label(mask)
    props = regionprops(lbl)
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]

        x2 = prop.bbox[3]
        y2 = prop.bbox[2]

        bboxes=[x1, y1, x2, y2]
        bboxes = bbox_to_xywh(bboxes)
    return bboxes

def getKeypointsVis(keypoints, frame_shape):
    """get keypoints for posetrack21
    """
    keypoints_vis = []
    for keypoint in keypoints:
        keypoints_vis.append(keypoint[0])
        keypoints_vis.append(keypoint[1])

        if validatePoint(keypoint, frame_shape):
            keypoints_vis.append(1)
        else:
            keypoints_vis.append(0)
    return keypoints_vis

def getCenterPoint(point1, point2):
    center = []
    center.append(point1[0] + point2[0] / 2)
    center.append(point1[1] + point2[1] / 2)
    return center

def getPossibleFaceArea(person, frame_shape):
    face_mask = None

    nose = person.getBodyPart('Nose')
    left_eye = person.getBodyPart('LeftEye')
    left_ear = person.getBodyPart('LeftEar')
    right_eye = person.getBodyPart('RightEye')
    right_ear = person.getBodyPart('RightEar')

    points_list = [nose, left_ear, left_eye, right_ear, right_eye]
    for i, point in enumerate(points_list):
        if validatePoint(point, frame_shape):
            points_list[i] = None
        else:
            points_list[i] = np.array(point)


    nose = points_list[0]
    left_eye = points_list[2]
    left_ear = points_list[1]
    right_eye = points_list[4]
    right_ear = points_list[3]
    radius_list = []
    radius = None
    if nose is not None:
        if right_ear is not None or left_ear is not None:
            if left_ear is not None:
                radius_list.append(getEuclideanDistance(nose, left_ear))
            if right_ear is not None:
                radius_list.append(getEuclideanDistance(nose, right_ear))
            radius = int(np.mean(radius_list))
        else:
            if left_eye is not None:
                radius_list.append(getEuclideanDistance(nose, left_eye))
            if right_eye is not None:
                radius_list.append(getEuclideanDistance(nose, right_eye))
                radius_mean = int(np.mean(radius_list))
                radius = radius_mean * 3
    elif right_ear is not None and left_ear is not None:
        radius = getEuclideanDistance(getCenterPoint(left_ear, right_ear), right_ear)
    elif right_eye is not None and left_eye is not None:
        radius = getEuclideanDistance(getCenterPoint(left_eye, right_eye), right_eye)
    elif right_eye is not None and right_ear is not None:
        radius = getEuclideanDistance(right_eye, right_ear)
    elif left_eye is not None and left_ear is not None:
        radius = getEuclideanDistance(right_eye, right_ear)
        
    if radius is not None:
        face_mask = np.zeros(frame_shape, np.float32)
        face_mask = cv2.circle(face_mask, nose, radius, 255, -1)
    return face_mask