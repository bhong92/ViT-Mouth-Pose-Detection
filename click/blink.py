import cv2 as cv
import mediapipe as mp
import time
import math
import numpy as np
import pyautogui
from threading import Thread

from . import utils



class Blink():
    def __init__(self):
        self.camstart = False
        self.frame = None
        blink_tread = Thread(target=self.blink_detect, args=())
        blink_tread.start()

    # landmark detection function 
    def landmarksDetection(self, img, results, draw=False):
        img_height, img_width= img.shape[:2]
        # list[(x,y), (x,y)....]
        mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
        if draw :
            [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

        # returning the list of tuples for each landmarks 
        return mesh_coord

    # Euclaidean distance 
    def euclaideanDistance(self, point, point1):
        x, y = point
        x1, y1 = point1
        distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
        return distance

    def left_blink_ratio(self, img, landmarks, left_indices):
        # LEFT_EYE 
        # horizontal line 
        lh_right = landmarks[left_indices[0]]
        lh_left = landmarks[left_indices[8]]

        # vertical line 
        lv_top = landmarks[left_indices[12]]
        lv_bottom = landmarks[left_indices[4]]

        lvDistance = self.euclaideanDistance(lv_top, lv_bottom)
        lhDistance = self.euclaideanDistance(lh_right, lh_left)

        leRatio = lhDistance/lvDistance
        return leRatio

    def right_blink_ratio(self, img, landmarks, right_indices):
        # Right eyes 
        # horizontal line 
        rh_right = landmarks[right_indices[0]]
        rh_left = landmarks[right_indices[8]]
        # vertical line 
        rv_top = landmarks[right_indices[12]]
        rv_bottom = landmarks[right_indices[4]]

        rhDistance = self.euclaideanDistance(rh_right, rh_left)
        rvDistance = self.euclaideanDistance(rv_top, rv_bottom)

        reRatio = rhDistance/rvDistance
        return reRatio

    def blink_detect(self):
        # variables 
        frame_counter =0
        CEF_COUNTER =0
        TOTAL_BLINKS =0
        leratio_max = float(0)
        leratio_min = float(10)
        reratio_max = float(0)
        reratio_min = float(10)
        # constants
        CLOSED_EYES_FRAME =3
        FONTS =cv.FONT_HERSHEY_COMPLEX
        TIME_BUFFER = 0.4
        CALIBRATION_TIME = 5
        time_buffer_start_time_left = 0
        time_buffer_start_time_right = 0
        blink_click_enabled = False

        # face bounder indices 
        FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

        # lips indices for Landmarks
        LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
        LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
        UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 
        # Left eyes indices 
        LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
        LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

        # right eyes indices
        RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
        RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]
        map_face_mesh = mp.solutions.face_mesh
        # camera object 
        # camera = cv.VideoCapture(0)
        with map_face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:

            # starting time here 
            start_time = time.time()
            # starting Video loop here.
            while True:
                if not self.camstart:
                    time.sleep(0.001)
                    start_time = time.time()
                    continue
                frame_counter +=1 # frame counter
                # ret, frame = camera.read() # getting frame from camera 
                frame = self.frame
                # if not ret: 
                #     break # no more frames break
                #  resizing frame
                
                frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
                frame_height, frame_width= frame.shape[:2]
                rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
                results  = face_mesh.process(rgb_frame)
                if results.multi_face_landmarks:
                    mesh_coords = self.landmarksDetection(frame, results, False)
                    leratio = self.left_blink_ratio(frame, mesh_coords, LEFT_EYE)
                    reratio = self.right_blink_ratio(frame, mesh_coords, RIGHT_EYE)
                    # dynamically update click threshold
                    leratio_max = max(leratio_max, leratio)
                    leratio_min = min(leratio_min, leratio)
                    leratio_click = (leratio_max*0.75 + leratio_min*0.25)
                    reratio_max = max(leratio_max, leratio)
                    reratio_min = min(leratio_min, leratio)
                    reratio_click = (reratio_max*0.75 + reratio_min*0.25)
                    if time.time() - start_time < CALIBRATION_TIME:
                        continue

                    # cv.putText(frame, f'ratio {ratio}', (100, 100), FONTS, 1.0, utils.GREEN, 2)
                    # utils.colorBackgroundText(frame,  f'Left Ratio : {round(leratio,2)}', FONTS, 0.7, (30,100),2, utils.PINK, utils.YELLOW)
                    # utils.colorBackgroundText(frame,  f'right Ratio : {round(reratio,2)}', FONTS, 0.7, (30,100),2, utils.PINK, utils.YELLOW)
                    # print(leratio,"----",reratio,"----",leratio_click,"----",reratio_click)
                    if leratio < leratio_click and reratio < reratio_click:
                        blink_click_enabled = True
                    
                    if leratio >leratio_click and blink_click_enabled:
                        if time_buffer_start_time_right==0:
                            if time_buffer_start_time_left == 0:
                                time_buffer_start_time_left = time.time()
                            elif time.time() - time_buffer_start_time_left > TIME_BUFFER:
                                time_buffer_start_time_left = 0
                                CEF_COUNTER +=1
                                # cv.putText(frame, 'Blink', (200, 50), FONTS, 1.3, utils.PINK, 2)
                                utils.colorBackgroundText(frame,  f'Left Clink', FONTS, 1.7, (int(frame_height/2), 100), 2, utils.YELLOW, pad_x=6, pad_y=6,)
                                pyautogui.click(button = 'left')
                                time.sleep(1)
                        elif time.time() - time_buffer_start_time_right < TIME_BUFFER:
                            blink_click_enabled = False
                            time_buffer_start_time_right = 0
                        
                    if reratio >reratio_click and blink_click_enabled:
                        if time_buffer_start_time_left==0:
                            if time_buffer_start_time_right == 0:
                                time_buffer_start_time_right = time.time()
                            elif time.time() - time_buffer_start_time_right > TIME_BUFFER:
                                time_buffer_start_time_right = 0
                                CEF_COUNTER +=1
                                # cv.putText(frame, 'Blink', (200, 50), FONTS, 1.3, utils.PINK, 2)
                                utils.colorBackgroundText(frame,  f'Right Clink', FONTS, 1.7, (int(frame_height/2), 100), 2, utils.YELLOW, pad_x=6, pad_y=6,)
                                pyautogui.click(button = 'right')  
                                time.sleep(1) 
                        elif time.time() - time_buffer_start_time_left < TIME_BUFFER:
                            blink_click_enabled = False
                            time_buffer_start_time_left = 0
                              
                            

                    else:
                        if CEF_COUNTER>CLOSED_EYES_FRAME:
                            TOTAL_BLINKS +=1
                            CEF_COUNTER =0
                    
                    cv.polylines(frame,  [np.array([mesh_coords[p] for p in LEFT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
                    cv.polylines(frame,  [np.array([mesh_coords[p] for p in RIGHT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)



                # calculating  frame per seconds FPS
                end_time = time.time()-start_time
                fps = frame_counter/end_time

                frame =utils.textWithBackground(frame,f'FPS: {round(fps,1)}',FONTS, 1.0, (30, 50), bgOpacity=0.9, textThickness=2)

                # cv.imshow('frame', frame)
                # key = cv.waitKey(2)
                # if key==ord('q') or key ==ord('Q'):
                #     break
            cv.destroyAllWindows()
            # camera.release()

if __name__ == '__main__':
    Blink()