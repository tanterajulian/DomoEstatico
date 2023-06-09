import cv2
import numpy as np
import time
import imutils
import tensorflow_hub as hub
import tensorflow as tf
import pandas as pd
from filterpy.kalman import KalmanFilter
import datetime

from matplotlib import pyplot as plt

# Function for stereo vision and depth estimation
import triangulation2 as tri
import calibration

# Mediapipe for face detection
import time


detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")
labels = pd.read_csv('./labels.csv',sep=';',index_col='ID')
labels = labels['OBJECT (2017 REL.)']

# Open both cameras
cap_right = cv2.VideoCapture(1)  #!Inicia camara derecha               
cap_left =  cv2.VideoCapture(0)  #!Inicia camara izquierda

cap_right.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap_left.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Stereo vision setup parameters
frame_rate = 30    #Camera frame rate (maximum at 120 fps)
B = 9               #Distance between the cameras [cm]
f = 7.8             #Camera lense's focal length [mm]
alpha = 68.6        #Camera field of view in the horisontal plane [degrees]

# Main program loop with face detector and depth estimation using stereo vision
#while(cap_right.isOpened() and cap_left.isOpened()): #! mientras que las camaras estan online
depth_arr = []

# Define the Kalman filter
kf = KalmanFilter(dim_x=1, dim_z=1)
kf.x = np.array([0])  # Initial state estimate
kf.F = np.array([[1]])  # State transition matrix
kf.H = np.array([[1]])  # Measurement matrix
kf.P = np.array([[1]])  # Covariance matrix of initial state estimate
kf.Q = np.array([[0.01]])  # Process noise covariance matrix
kf.R = np.array([[1]])  # Measurement noise covariance matrix

depth2_filtered = []

while(True):

    succes_right, frame_right = cap_right.read() #! lee los cuadros de las camaras
    succes_left, frame_left = cap_left.read()

################## CALIBRATION #########################################################

    frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left) #! Rectifica las imagenes

########################################################################################

    # If cannot catch any frame, break
    if not succes_right or not succes_left:             
        break

    else:
        
        start = time.time()
        
        # Convert the BGR image to RGB
        frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
        frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)

        # Process the image and find faces #!DETECCION

        #Is optional but i recommend (float convertion and convert img to tensor image)
        frame_right_tensor = tf.convert_to_tensor(frame_right, dtype=tf.uint8)
        frame_left_tensor = tf.convert_to_tensor(frame_left, dtype=tf.uint8)

        #Add dims to rgb_tensor
        frame_right_tensor = tf.expand_dims(frame_right_tensor , 0)
        frame_left_tensor = tf.expand_dims(frame_left_tensor , 0)
        
        boxes_r, scores_r, classes_r, num_detections_r = detector(frame_right_tensor)
        boxes_l, scores_l, classes_l, num_detections_l = detector(frame_left_tensor)
        
        
        pred_labels_r = classes_r.numpy().astype('int')[0]
        pred_labels_l = classes_l.numpy().astype('int')[0]

        
        pred_labels_r = [labels[i] for i in pred_labels_r]
        pred_labels_l = [labels[i] for i in pred_labels_l]

        pred_boxes_r = boxes_r.numpy()[0].astype('int')
        pred_boxes_l = boxes_l.numpy()[0].astype('int')
        
        pred_scores_r = scores_r.numpy()[0]
        pred_scores_l = scores_l.numpy()[0]


        # Convert the RGB image to BGR
        frame_right = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)
        frame_left = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)
        

        ################## CALCULATING DEPTH #########################################################

        center_point_right = 0
        center_point_left = 0

        for (score_r, (ymin_r, xmin_r, ymax_r, xmax_r), label_r), (score_l, (ymin_l, xmin_l, ymax_l, xmax_l), label_l) in zip(zip(pred_scores_r, pred_boxes_r, pred_labels_r), zip(pred_scores_l, pred_boxes_l, pred_labels_l)):
            
            score_txt = f'{100 * round(score_r,0)}'
            if label_r == "person" and score_r >= 0.3:
                nobody = 0
                img_boxes = cv2.rectangle(frame_right, (xmin_r, ymax_r), (xmax_r, ymin_r), (0, 255, 0), 1)
                center_point_right = ((xmax_r + xmin_r) / 2, (ymax_r + ymin_r) / 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img_boxes, label_r, (xmin_r, ymax_r - 10), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            
            score_txt = f'{100 * round(score_l,0)}'
            if label_l == "person" and score_l >= 0.3:
                nobody = 0
                img_boxes = cv2.rectangle(frame_left, (xmin_l, ymax_l), (xmax_l, ymin_l), (0, 255, 0), 1)
                center_point_left = ((xmax_l + xmin_l) / 2, (ymax_l + ymin_l) / 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img_boxes, label_l, (xmin_l, ymax_l - 10), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(img_boxes, score_txt, (xmax_l, ymax_l - 10), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)


        # Function to calculate depth of object. 
        if center_point_left != 0 and center_point_right != 0:
            print("punto ", center_point_right)
            print("punto ", center_point_left)
            depth = tri.find_depth(center_point_right, center_point_left, frame_right, frame_left, B, f, alpha)

            depth_cm = depth*100

            # Update the Kalman filter with the measurement
            kf.predict()
            kf.update(depth_cm)

            depth_arr.append(depth_cm)

            # Get the filtered estimate of depth2
            depth2_filtered.append(kf.x[0])

            print("DepthKalman: ", kf.x[0])

            #Si la distancia medida es menor a 200, imprimimos en un archivo log 

            if kf.x[0] < 200:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open('/home/nvidia/DomoEstatico/DomoPC/logDomo.txt', 'w') as file:
                    file.write(f"Time: {current_time}, Distancia a la persona: {kf.x[0]}\n")

                # Capture images from each camera
                ret, left_image = cap_left.read()
                ret, right_image = cap_right.read()

                # Save the images (overwrite the existing images)
                left_image_path = "/home/nvidia/DomoEstatico/DomoPC/left_image.jpg"
                right_image_path = "/home/nvidia/DomoEstatico/DomoPC/right_image.jpg"

                cv2.imwrite(left_image_path, left_image)
                cv2.imwrite(right_image_path, right_image)

            
            # Calculo de distancia teniendo en cuenta ultimas 10 mediciones

            # if len(depth_arr) > 10:
            #     depth_arr.pop(0)                #borramos el primer valor que se encuentre en el array
            
            # depth_prom = np.mean(depth_arr)     #calculamos el promedio de las mediciones existentes en el array
          
            # cv2.putText(frame_right, "Distance: " + str(round(depth_prom,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
            # cv2.putText(frame_left, "Distance: " + str(round(depth_prom,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)

            # print("DepthPromedio:", depth_prom)
        
        else:
            cv2.putText(frame_left, "Area Clear", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
            cv2.putText(frame_right, "Area Clear", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
            print("no hay nadie")

        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime
        #print("FPS: ", fps)

        cv2.putText(frame_right, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
        cv2.putText(frame_left, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)                                   


        # Show the frames
        cv2.imshow("frame right", frame_right) 
        cv2.imshow("frame left", frame_left)


        # Hit "q" to close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Release and destroy all windows before termination
cap_right.release()
cap_left.release()

cv2.destroyAllWindows()
