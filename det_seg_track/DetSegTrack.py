
from det_seg_track.utils import ImageAnnotator
from det_seg_track.tracker import Tracker
from det_seg_track.segmentator import Segmentator
from det_seg_track.detector import Detector

class DetSegTrack:
    """
    Detects, tracks and segments joints.

    Params:
        detector_name: detector,
        tracker_name: tracker,
        segmentator_name: segm,
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    def __init__(self, detector_name, tracker_name, segmentator_name = None, use_deployed_model = True, validate_person = True):
        self.detector_name = detector_name
        self.tracker_name = tracker_name
        self.segmentator_name = segmentator_name
        self.validate_person = validate_person

        # detector
        self.detector_model = Detector(detector_name, use_deployed_model = use_deployed_model)

        # tracker
        self.detect_and_track = False
        self.tracker = None
        yolov_trackers = ['botsort', 'bytetrack']
        if ('yolov' in detector_name and tracker_name in yolov_trackers):
            self.tracker_name = tracker_name + '.yaml'
            self.detect_and_track = True
        elif (tracker_name == 'bytetrack'):
            self.tracker = Tracker(tracker_name)
            self.detect_and_track = False

        # segmentation
        if self.segmentator_name is not None:
            self.seg_model = Segmentator(segmentator_name)
        else:
            self.seg_model = segmentator_name

    def estimate(self, frame, visualize = True):
        annotated_frame, person_results, params = self.detector_model.useDetector(frame, self.detect_and_track, 
                                                                                  self.tracker_name, validatePerson = self.validate_person)
        if (not self.detect_and_track):
            person_results = self.tracker.useTracker(person_results, frame)        

        if self.segmentator_name is not None:
            if ('FastSAM' in self.segmentator_name):
                self.seg_model.initFastSAM(frame)
            for person in person_results:
                person.mask = self.seg_model.getFaceMask(frame, person)
        
        return annotated_frame, person_results, params
    
    def __del__(self):
        del self.detector_model
        if self.tracker is not None:
            del self.tracker
        if self.seg_model is not None:
            del self.seg_model