import cv2
import numpy as np

def BasicAlgorithm(face, format = 'BGR'):
    if format == 'BGR':
        component_G = face[:,:,1]
        return np.sum(component_G)