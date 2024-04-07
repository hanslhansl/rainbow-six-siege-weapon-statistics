import cv2
import numpy as np

vid = cv2.VideoCapture(r"E:\Paul\Videos\Radeon ReLive\Tom Clancy's Rainbow Six Siege\Tom Clancy's Rainbow Six Siege_replay_2024.04.06-16.48.mp4")


# Read the entire file until it is completed 
while(vid.isOpened()): 
    ret, frame = vid.read() 
    if ret == True:
        cv2.imshow('Frame', frame[1000:1400, 600:900])
        if cv2.waitKey(25) & 0xFF == ord('q'): 
            break
    else: 
        break
