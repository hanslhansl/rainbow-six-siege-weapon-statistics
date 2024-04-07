import cv2, os, sys
import numpy as np

p = sys.argv[1]
#p = r"E:\Paul\Videos\Radeon ReLive\Tom Clancy's Rainbow Six Siege\Tom Clancy's Rainbow Six Siege_replay_2024.04.06-21.58.mp4"

vid = cv2.VideoCapture(p)
frame_rate = vid.get(cv2.CAP_PROP_FPS)
print("frame rate", frame_rate)

i = 0
state = 0   # 0: nothing, 1: white dot, 2: acog, 3: red dot C
last_dot = 0
while(vid.isOpened()):
    ret, frame = vid.read()
    if ret != True:
        break
    
    if all([np.allclose(frame[719+y, 1279+x], [0xFF, 0xFF, 0xFF], 0., 40.) for y, x in np.ndindex((2, 2))]):
        if state != 1:
            state = 1
        last_dot = i
        
    elif all([np.allclose(frame[726+y, 1279+x], [0x21, 0x00, 0xDB], 0., 40.) for y, x in np.ndindex((5, 3))]):  # acog
        if state != 2:
            print(f"acog [{(i - last_dot - 2)}, {i - last_dot}] fps = [{(i - last_dot - 2) / frame_rate}, {(i - last_dot) / frame_rate}] s")
            state = 2
            
    elif all([np.allclose(frame[719+y, 1279+x], [0x27, 0x00, 0xE6], 0., 40.) for y, x in np.ndindex((2, 2))]):# red dot C
        if state != 3:
            print(f"red dot C [{(i - last_dot - 2)}, {i - last_dot}] fps = [{(i - last_dot - 2) / frame_rate}, {(i - last_dot) / frame_rate}] s")
            state = 3

    else:
        if state != 0:
            state = 0
            
    i += 1

input("\nCompleted!")

