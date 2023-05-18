# Open cv is used for image recognition
import cv2

# mediapipe is used for eye movement.
import mediapipe as mp

# pyautogui is automation module which is used to control keyboard and mouse.
import pyautogui

# VideoCapture() method is used for reading the visuals through camera. Parameter '0' is used for using the 1st web camera that is available
cam = cv2.VideoCapture(0)

# face mesh is the 3D model of the face which creates a mesh of your face. It is a part of solution of media pipe
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

screen_w, screen_h = pyautogui.size()

# Since we are capturing video so we need to read frames recursively, this is the reason we are using while loop.
while True:
    # We read frames from the video.
    _, frame = cam.read()

    #
    frame = cv2.flip(frame, 1)

    # cvtColor() method will convert gray scale frame into different color.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Creates a output from the RGB frame from the face mesh which we used.
    output = face_mesh.process(rgb_frame)

    # Whenever detecting face, creates all the landmarks on the face for eg. nose, eye, etc. output.multi_face_landmarks will return the array of landmarks on our face.
    landmarks_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    # if landmarks_points are not null then we want to go further.
    if landmarks_points:

        # Since we want 1st face's landmark so we extract the 1st value from the landmarks_points
        landmarks = landmarks_points[0].landmark
        
        # for click operation.landmark values are predefined
        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            # for detecting x and y axis in landmark and then drawing circle
            x1 = int(landmark.x * frame_w)
            y1 = int(landmark.y * frame_h)
            # color will be yellow
            cv2.circle(frame, (x1, y1), 3, (0, 255, 255))
        # for vertical position y axis. for overlapping the landmarks
        if (left[0].y - left[1].y) < 0.01:
            pyautogui.click(button = 'left')   
            pyautogui.sleep(1)                           #

        right = [landmarks[374], landmarks[386]]
        for landmark in right:
            # for detecting x and y axis in landmark and then drawing circle
            x2 = int(landmark.x * frame_w)
            y2 = int(landmark.y * frame_h)
            # color will be yellow
            cv2.circle(frame, (x2, y2), 3, (255, 255, 255))
        # for vertical position y axis. for overlapping the landmarks
        if (right[0].y - right[1].y) < 0.01:
            pyautogui.click(button = 'right')        
            pyautogui.sleep(1)    

    # imshow() it tells cv2 to show the image.
    cv2.imshow('Eye Control Mouse', frame)

    # waitKey() method uses to open the web camera
    cv2.waitKey(1)