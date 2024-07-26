import numpy as np
import cv2

BODY_PART_MAPPING = {
    "COCO" : {
        "Nose" : 0,
        "RightShoulder" : 6,
        "LeftShoulder" : 5,
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
        return "Test()"
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

        # validation
        bbox_area = (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
        image_area = (frame_shape[0] * frame_shape[1])

        # if bbox is smaller than 20% of image, it's not valid
        is_bbox_valid = (bbox_area/image_area) > 0.2 and self.bbox[4] > conf

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
    if (point[0] <= frame_shape[0] and point[1] <= frame_shape[1]):
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
    print(points)
    labels.append(1)

    if include_negative:
        for part in negative_points:
            point = person.getBodyPart(part, reverse = reverse)
            if validatePoint(point, frame_shape):
                points.append(point)
                labels.append(0)

    return points, labels

