import numpy as np
from typing import List, Dict
from ByteTrack.yolox.tracker.byte_tracker import STrack, BYTETracker
from det_seg_track.utils import bbox_ious
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
        self.previous_detections = None
        self.tracks_mapping = {
            'person_id': [],
            'track_id': []
        }
        if (tracker_name == 'bytetrack'):
            self.tracker = BYTETracker(BYTETrackerArgs())

    def useTracker(self, person_results, frame):
        if (self.tracker_name == 'bytetrack'):
            if len(person_results) > 0:
                tracks = self.tracker.update(
                    output_results=detections2boxes(person_results),
                    img_info=frame.shape,
                    img_size=frame.shape)
                if self.previous_detections is not None:
                    _, self.tracks_mapping = match_detections_with_tracks(self.previous_detections, tracks, self.tracks_mapping, True)
                person_results, self.tracks_mapping = match_detections_with_tracks(person_results, tracks, self.tracks_mapping, False)
                self.previous_detections = person_results
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
    tracks: List[STrack],
    tracks_mapping: Dict,
    is_previous: bool
) -> List:
    """
    match current tracks with previous detections, 
    assign new ids from previous detection to current matches if possible


    match current tracks with current detections, add new ids if no track matches detection
    """
    detection_boxes = detections2boxes(detections=detections, with_confidence=False)
    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = bbox_ious(tracks_boxes, detection_boxes)
    track2detection = np.argmax(iou, axis=1)
    print('detections ', detection_boxes.shape)
    print('tracks ', tracks_boxes.shape)
    print('tracks raw ', tracks)
    print('track2detection shape ', track2detection.shape)
    print('track2detection ', track2detection)
    print('mapping ', tracks_mapping)
    print('det ', detections)

    # assure that tracker and detection can be selected
    if track2detection.shape[0] <= len(tracks) and track2detection.shape[0] <= len(detections):
        for tracker_index, detection_index in enumerate(track2detection):
            if iou[tracker_index, detection_index] != 0:
                track_id = tracks[tracker_index].track_id
                det_id = detections[detection_index].tracker_id

                if det_id is None:
                    det_id = track_id

                # if track id is new, map it to matched person id
                # person id can come from previous frame and doesnt have to be identical to track id
                if track_id not in tracks_mapping['track_id'] and det_id not in tracks_mapping['person_id']:
                    tracks_mapping['track_id'].append(track_id)
                    tracks_mapping['person_id'].append(det_id)
                elif track_id not in tracks_mapping['track_id'] and track_id not in tracks_mapping['person_id']:
                    tracks_mapping['track_id'].append(track_id)
                    tracks_mapping['person_id'].append(track_id)     
                
                # assign correct person id mapped to current track id
                track_idx = tracks_mapping['track_id'].index(track_id)
                detections[detection_index].tracker_id = tracks_mapping['person_id'][track_idx]

        # assure no None tracker_ids are present, add new ones if neccessary
        if not is_previous:
            max_id = max(tracks_mapping['person_id'])
            for detection in detections:
                if detection.tracker_id is None:
                    max_id += 1
                    detection.tracker_id = max_id
                #tracks_mapping['person_id'].append(max_id)
    else:
        max_id = max(tracks_mapping['person_id'])
        for detection in detections:
            if detection.tracker_id is None:
                max_id += 100
                detection.tracker_id = max_id
                
    print('det after track', detections)
            
    return detections, tracks_mapping
