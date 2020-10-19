# @Author: ASHISH SASMAL <ashish>
# @Date:   16-10-2020
# @Last modified by:   ashish
# @Last modified time: 19-10-2020

import cv2
import numpy as np
import imutils
from live_cam import live
from sklearn.metrics.pairwise import euclidean_distances as eqd
import math

fgbg = cv2.createBackgroundSubtractorMOG2(history=20)

bg = None

def contours(img2):
    cnts = cv2.findContours(img2.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return cnts

def bgextract(frame, aWeight=0.5):
    global bg
    # initialize the background
    if bg is None:
        bg = frame.copy().astype("float")
        return
    cv2.accumulateWeighted(frame, bg, aWeight)

def subtract(frame):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), frame)
    thresholded = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((5,5))
    erosion = cv2.erode(thresholded, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    cnts,h = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)


# cv2.namedWindow("MOG",cv2.WINDOW_NORMAL)
top, right, bottom, left = 0, 265, 260, 0
num_frames = 0
# cap = cv2.VideoCapture(0)

while True:
    # ret, frame = cap.read()
    frame = live()
    # print(frame.shape)
    blur = cv2.GaussianBlur(frame, (3,3), 0)
    count =0
    clone=frame.copy()
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # print(num_frames)

    roi = frame[top:bottom, left:right]

    if num_frames<30:

        blur_roi = cv2.GaussianBlur(roi, (3,3), 0)
        # print(blur_roi.shape)
        bgextract(blur_roi)
    else:
        hand = subtract(roi)
        # print("here")
        if hand is not None:

            (thresholded, segmented) = hand
            hull = cv2.convexHull(segmented,returnPoints = False)
            if len(hull)>3:
                defects = cv2.convexityDefects(segmented,hull)
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s,e,f,d = defects[i,0]
                        start = tuple(segmented[s][0])
                        end = tuple(segmented[e][0])
                        far = tuple(segmented[f][0])
                        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                        if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                            count += 1
                            cv2.circle(thresholded, far, 8, [211, 84, 0], -1)
            # roi = cv2.resize(roi, (200,200))
            print(count)
            cv2.imshow("Thresh",thresholded)

    fgmask = cv2.resize(frame, (720,360))
    cv2.rectangle(fgmask, (left, top), (right, bottom), (0,255,0), 2)
    cv2.imshow("MOG",fgmask)
    num_frames+=1

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
