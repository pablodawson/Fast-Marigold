import cv2
import numpy as np

def warp_with_flow(flow, curImg):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    nextImg = cv2.remap(curImg, flow, None, cv2.INTER_LINEAR)
    return nextImg