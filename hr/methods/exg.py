import numpy as np

def ExG(face, format = 'BGR'):
    if format == 'BGR':
        component_G = face[:,:,1]
        component_B = face[:,:,0]
        component_R = face[:,:,2]
        return np.sum(component_G)