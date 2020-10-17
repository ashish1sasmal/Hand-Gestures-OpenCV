# @Author: ASHISH SASMAL <ashish>
# @Date:   01-10-2020
# @Last modified by:   ashish
# @Last modified time: 16-10-2020

import cv2
import numpy as np

bg = None
fgbg = cv2.createBackgroundSubtractorMOG2(history=20)

def subtract(frame):    
    fgmask = fgbg.apply(frame)
    kernel = np.ones((9,9), np.uint8)
    ers = cv2.erode(fgmask,kernel,iterations=1)
    fgmask = cv2.resize(ers, (720,360))
    return fgmask
