import numpy as np
from typing import List
from ByteTrack.yolox.tracker.byte_tracker import STrack, BYTETracker
from onemetric.cv.utils.iou import box_iou_batch
"""
BYTETracker does not assign tracker_id to existing bounding boxes but rather
predicts the next bounding box position based on previous one. Therefore, we 
need to find a way to match our bounding boxes with predictions.

usage example:

byte_tracker = BYTETracker(BYTETrackerArgs())
for frame in frames:
    ...
    results = model(frame, size=1280)
    detections = Detection.from_results(
        pred=results.pred[0].cpu().numpy(), 
        names=model.names)
    ...
    tracks = byte_tracker.update(
        output_results=detections2boxes(detections=detections),
        img_info=frame.shape,
        img_size=frame.shape)
    detections = match_detections_with_tracks(detections=detections, tracks=tracks)
"""
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

class Tracker:
    def __init__(self, tracker_name):
        self.tracker_name = tracker_name
        if (tracker_name == 'bytetrack'):
            self.tracker = BYTETracker(BYTETrackerArgs())

    def useTracker(self, person_results, frame):
        if (self.tracker_name == 'bytetrack'):
            if len(person_results) > 0:
                tracks = self.tracker.update(
                    output_results=detections2boxes(person_results),
                    img_info=frame.shape,
                    img_size=frame.shape)
                person_results = match_detections_with_tracks(person_results, tracks)
        return person_results
        

# converts List[Detection] into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: List, with_confidence: bool = True) -> np.ndarray:
    return np.array([
            detection.bbox
        if with_confidence else
            detection.bbox[:-1]
        for detection
        in detections
    ], dtype=np.float32)


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=np.float32)


# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: List, 
    tracks: List[STrack]
) -> List:
    detection_boxes = detections2boxes(detections=detections, with_confidence=False)
    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detection_boxes)
    track2detection = np.argmax(iou, axis=1)
    
    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            detections[detection_index].tracker_id = tracks[tracker_index].track_id
    return detections
