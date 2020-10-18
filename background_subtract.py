# @Author: ASHISH SASMAL <ashish>
# @Date:   16-10-2020
# @Last modified by:   ashish
# @Last modified time: 18-10-2020

import cv2
import numpy as np
import imutils
from live_cam import live

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

    cnts,h = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)


cv2.namedWindow("MOG",cv2.WINDOW_NORMAL)
top, right, bottom, left = 0, 265, 260, 0
num_frames = 0
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    # frame = live()
    clone=frame.copy()
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if num_frames<30:
        bgextract(frame[top:bottom, left:right])
    else:
        hand = subtract(frame[top:bottom, left:right])
        if hand is not None:
            (thresholded, segmented) = hand
            c = cv2.convexHull(segmented)
            # print(c)
            thresholded = cv2.cvtColor(thresholded,cv2.COLOR_GRAY2BGR)
            # extLeft = tuple(c[c[:, :, 0].argmin()][0])
            # extRight = tuple(c[c[:, :, 0].argmax()][0])
            # extTop = tuple(c[c[:, :, 1].argmin()][0])
            # extBot = tuple(c[c[:, :, 1].argmax()][0])
            # cv2.circle(thresholded, extLeft, 8, (0, 0, 255), -1)
            # cv2.circle(thresholded, extRight, 8, (0, 255, 0), -1)
            # cv2.circle(thresholded, extTop, 8, (255, 0, 0), -1)
            # cv2.circle(thresholded, extBot, 8, (255, 255, 0), -1)
            # cx = (extLeft[0]+extRight[0])//2
            # cy = (extTop[1]+extBot[1])//2
            # cv2.circle(thresholded, (cx,cy), 8, (255, 255, 0), -1)
            for i in c:
                cv2.circle(thresholded, tuple(i[0]), 2, (255, 255, 0), -1)
            # cv2.drawContours(thresholded, [ch ], -1, (0, 255, 0))
            cv2.imshow("Theshold", thresholded)

    fgmask = cv2.resize(frame, (720,360))
    cv2.rectangle(fgmask, (left, top), (right, bottom), (0,255,0), 2)
    cv2.imshow("MOG",fgmask)
    num_frames+=1

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
