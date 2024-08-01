
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
    def __init__(self, detector_name, tracker_name, segmentator_name, use_deployed_model = True):
        self.detector_name = detector_name
        self.tracker_name = tracker_name
        self.segmentator_name = segmentator_name

        # detector
        self.detector_model = Detector(detector_name, use_deployed_model = use_deployed_model)

        # tracker
        yolov_trackers = ['botsort', 'bytetrack']
        if ('yolov' in detector_name and tracker_name in yolov_trackers):
            self.tracker_name = tracker_name + '.yaml'
            self.detect_and_track = True
        elif (tracker_name == 'bytetrack'):
            self.tracker = Tracker(tracker_name)
            self.detect_and_track = False

        # segmentation
        self.seg_model = Segmentator(segmentator_name)

    def estimate(self, frame):
        annotated_frame, person_results, params = self.detector_model.useDetector(frame, self.detect_and_track, 
                                                                                  self.tracker_name)
        if (not self.detect_and_track):
            person_results = self.tracker.useTracker(person_results, frame)        

        annotator = ImageAnnotator(frame, annotated_frame, convertRGBToBGR = params['convertRGBToBGR'])

        if ('FastSAM' in self.segmentator_name):
            self.seg_model.initFastSAM(frame)
        for person in person_results:
            person.mask = self.seg_model.getFaceMask(frame, person)
            annotator.annotateImage(person, showTracker = params['show_tracker'])
        
        return annotator.annotated_frame, person_results