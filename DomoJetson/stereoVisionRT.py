import cv2
import numpy as np

from jetson_inference import detectNet
from jetson_utils import videoOutput, videoSource, cudaAllocMapped, cudaConvertColor, cudaDeviceSynchronize, cudaToNumpy, cudaFromNumpy

import sys
import argparse
import datetime

from calibration_store import load_stereo_coefficients

log_path = '/home/nvidia/DomoEstatico/DomoJetson/logDomo.txt'

left_image_path = "/home/nvidia/DomoEstatico/DomoJetson/left_image.jpg"
right_image_path = "/home/nvidia/DomoEstatico/DomoJetson/right_image.jpg"

parser = argparse.ArgumentParser()
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use")
parser.add_argument('--calibration_file', type=str, required=True, help='Path to the stereo calibration file')


is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

input0 = videoSource('/dev/video0', argv=sys.argv)
input1 = videoSource('/dev/video1', argv=sys.argv)
output0 = videoOutput(args.output, argv=sys.argv+is_headless)
output1 = videoOutput(args.output, argv=sys.argv+is_headless)

net = detectNet(args.network, sys.argv, args.threshold)

K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q = load_stereo_coefficients(args.calibration_file)  # Get cams params


while(True):
    img0 = input0.Capture()
    img1 = input1.Capture()

    # convert to BGR, since that's what OpenCV expects
    bgr_img_0 = cudaAllocMapped(width=img0.width,
                            height=img0.height,
                            format='bgr8')
    bgr_img_1 = cudaAllocMapped(width=img1.width,
                            height=img1.height,
                            format='bgr8')

    cudaConvertColor(img0, bgr_img_0)
    cudaConvertColor(img1, bgr_img_1)

    # make sure the GPU is done work before we convert to cv2
    cudaDeviceSynchronize()

    # convert to cv2 image (cv2 images are numpy arrays)
    frame_right = cudaToNumpy(bgr_img_0)
    frame_left = cudaToNumpy(bgr_img_1)


################## CALIBRATION #########################################################
    height, width, channel = frame_left.shape  # We will use the shape for remap

    # Undistortion and Rectification part!
    leftMapX, leftMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (width, height), cv2.CV_32FC1)
    left_rectified = cv2.remap(frame_left, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    rightMapX, rightMapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (width, height), cv2.CV_32FC1)
    right_rectified = cv2.remap(frame_right, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

    left_rectified_bgr = cudaFromNumpy(left_rectified, isBGR=True)

    # convert from BGR -> RGB
    left_rectified = cudaAllocMapped(width=left_rectified_bgr.width,
                            height=left_rectified_bgr.height,
                            format='rgb8')

    cudaConvertColor(left_rectified_bgr, left_rectified)

    right_rectified_bgr = cudaFromNumpy(right_rectified, isBGR=True)

    # convert from BGR -> RGB
    right_rectified = cudaAllocMapped(width=right_rectified_bgr.width,
                            height=right_rectified_bgr.height,
                            format='rgb8')

    cudaConvertColor(right_rectified_bgr, right_rectified)

    # If cannot catch any frame, break
    if not input0.IsStreaming() or not input1.IsStreaming():             
        break

    else:
        
        detections_right = net.Detect(right_rectified, overlay=args.overlay)
        detections_left = net.Detect(left_rectified, overlay=args.overlay)

        center_point_right = 0
        center_point_left = 0

        for detection in detections_right:
            if detection.ClassID == 1:
                center_point_right = detection.Center

        for detection in detections_left:
            if detection.ClassID == 1:
                center_point_left = detection.Center

        # Function to calculate depth of object. 
        if center_point_left != 0 and center_point_right != 0:
 
            point_3d_homog = cv2.triangulatePoints(P1, P2, center_point_left, center_point_right)

            point_3d_cartesian = point_3d_homog[:3] / point_3d_homog[3]

            distance = np.linalg.norm(point_3d_cartesian)
            print(distance)

            #Si la distancia medida es menor a 200, imprimimos en un archivo log 

            if distance < 5.0:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(log_path, 'w') as file:
                    file.write(f"Time: {current_time}, Distancia a la persona: {distance}\n")

                # Capture images from each camera
                img0 = input0.Capture()
                img1 = input1.Capture()

                # Convert to numpy arrays
                # Convert to numpy arrays and convert color format to BGR
                left_image = cv2.cvtColor(cudaToNumpy(img0), cv2.COLOR_RGB2BGR)
                right_image = cv2.cvtColor(cudaToNumpy(img1), cv2.COLOR_RGB2BGR)

                # Save the images (overwrite the existing images)
            
                cv2.imwrite(left_image_path, left_image)
                cv2.imwrite(right_image_path, right_image)

        
        else:
            print("no hay nadie")

        output0.Render(left_rectified)
        output0.SetStatus("{:s} | Network {:.0f} FPS".format("Left", net.GetNetworkFPS()))
        output1.Render(right_rectified)
        output1.SetStatus("{:s} | Network {:.0f} FPS".format("Right", net.GetNetworkFPS()))
        
