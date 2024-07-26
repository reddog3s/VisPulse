
from ultralytics import YOLO
import os
from mmpose.apis import MMPoseInferencer
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
import torch
from utils import Person
import cv2

class Detector:
    def __init__(self, detector_name, device = 'cpu', use_deployed_model = True):
        self.detector_name = detector_name
        self.use_deployed_model = use_deployed_model
        if (detector_name == 'yolov8l-pose'):
            detector_path = os.path.join('./checkpoints', detector_name + '.pt')
            self.detector_model = YOLO(detector_path)
        elif (detector_name == 'yolov8m-pose'):
            detector_path = os.path.join('./checkpoints', detector_name + '.pt')
            self.detector_model = YOLO(detector_path)
        elif (detector_name == 'yolov8n-pose'):
            detector_path = os.path.join('./checkpoints', detector_name + '.pt')
            self.detector_model = YOLO(detector_path)
        elif (detector_name == 'yolov8s-pose'):
            detector_path = os.path.join('./checkpoints', detector_name + '.pt')
            self.detector_model = YOLO(detector_path)
        if (detector_name == 'rtmo-l'):
            if use_deployed_model:
                deploy_cfg = './mmdeploy/configs/mmpose/pose-detection_rtmo_onnxruntime_dynamic.py'
                model_cfg = './mmpose/configs/body_2d_keypoint/rtmo/body7/rtmo-l_16xb16-600e_body7-640x640.py'
                backend_model =[ os.path.join('./deployed_models', detector_name, 'end2end.onnx') ]

                # read deploy_cfg and model_cfg
                deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)
                self.deploy_cfg = deploy_cfg
                # build task and backend model
                self.task_processor = build_task_processor(model_cfg, deploy_cfg, device)
                self.detector_model = self.task_processor.build_backend_model(backend_model)
                # detector_path = os.path.join('./deployed_models', detector_name)
                # self.detector_model = PoseDetector(
                #    model_path=detector_path, device_name=device, device_id=0)
            else:
                detector_path = os.path.join('./checkpoints', 
                                            'rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.pth')
                detector_config = './mmpose/configs/body_2d_keypoint/rtmo/body7/rtmo-l_16xb16-600e_body7-640x640.py'
                self.detector_model = MMPoseInferencer(
                    pose2d=detector_config,
                    pose2d_weights=detector_path,
                    device=device
                )
        if (detector_name == 'rtmo-l-crowdpose'):
            if use_deployed_model:
                deploy_cfg = './mmdeploy/configs/mmpose/pose-detection_rtmo_onnxruntime_dynamic.py'
                model_cfg = 'rtmo-l_16xb16-700e_body7-crowdpose-640x640.py'
                backend_model = [ os.path.join('./deployed_models', detector_name, 'end2end.onnx') ]

                # read deploy_cfg and model_cfg
                deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)
                self.deploy_cfg = deploy_cfg
                # build task and backend model
                self.task_processor = build_task_processor(model_cfg, deploy_cfg, device)
                self.detector_model = self.task_processor.build_backend_model(backend_model)
                # detector_path = os.path.join('./deployed_models', detector_name)
                # self.detector_model = PoseDetector(
                #    model_path=detector_path, device_name=device, device_id=0)
            else:
                detector_path = os.path.join('./checkpoints', 
                                            'rtmo-l_16xb16-700e_body7-crowdpose-640x640-5bafdc11_20231219.pt')
                detector_config = './mmpose/configs/body_2d_keypoint/rtmo/crowdpose/rtmo-l_16xb16-700e_body7-crowdpose-640x640.py'
                self.detector_model = MMPoseInferencer(
                    pose2d=detector_config,
                    pose2d_weights=detector_path,
                    device=device
                )

    def useDetector(self, frame, detect_and_track = False, tracker_name = None):
        if ('yolo' in self.detector_name):
            if (detect_and_track):
                results = self.detector_model.track(frame, persist=True, tracker=tracker_name)
            else:
                results = self.detector_model(frame)

            annotated_frame = results[0].plot()
            show_tracker = False
            convertRGBToBGR = False

            person_results = []
            for i, box in enumerate(results[0].boxes.xyxy):
                person = Person()
                
                person.fromUltralyticsResult(results[0].keypoints.xy[i].tolist(),
                                                box.tolist(),
                                                results[0].boxes.conf.tolist()[i],
                                                results[0].boxes.id.tolist()[i])
                if (person.validatePerson(frame.shape)):
                    person_results.append(person)
        else:
            if self.use_deployed_model:
                # process input image
                input_shape = get_input_shape(self.deploy_cfg)
                model_inputs, _ = self.task_processor.create_input(frame, input_shape)

                # do model inference
                with torch.no_grad():
                    results = self.detector_model.test_step(model_inputs)
                
                annotated_frame = frame
                person_results = []
                bbox_scores = results[0].pred_instances.bbox_scores.cpu().detach().numpy()
                keypoints_scores = results[0].pred_instances.keypoint_scores.cpu().detach().numpy()
                
                for i, keypoints in enumerate(results[0].pred_instances.keypoints):
                    person = Person()
                    person.fromMMDeployResult(keypoints, 
                                              results[0].pred_instances.bboxes[i], 
                                              bbox_scores[i])
                    if (person.validatePerson(annotated_frame.shape, keypoints_conf = keypoints_scores[i])):
                        person_results.append(person)
                        for [x, y] in person.keypoints.astype(int):
                            annotated_frame = cv2.circle(annotated_frame, (x, y), 20, (0, 255, 0), -1)

                show_tracker = True
                convertRGBToBGR = False
            else:
                result_generator = self.detector_model(frame, return_vis=True, thickness=3, draw_bbox=True)
                results = next(result_generator)
                annotated_frame = results['visualization'][0]

                person_results = []
                for instance in results['predictions'][0]:
                    person = Person()
                    person.fromMMPoseResult(instance)
                    if (person.validatePerson(annotated_frame.shape)):
                        person_results.append(person)
            
                show_tracker = True
                convertRGBToBGR = True
        
        params = {
            'show_tracker' : show_tracker,
            'convertRGBToBGR' : convertRGBToBGR
        }

        return annotated_frame, person_results, params