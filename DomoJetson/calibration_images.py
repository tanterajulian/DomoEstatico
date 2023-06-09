import cv2

dev_L = 0
dev_R = 1
width = 640
height = 480
fps = 30
# Open both cameras
gst_str_L = ('v4l2src device=/dev/video{} ! '
               'video/x-raw, width=(int){}, height=(int){} ! '
               'videoconvert ! appsink').format(dev_L, width, height, fps)
gst_str_R = ('v4l2src device=/dev/video{} ! '
               'video/x-raw, width=(int){}, height=(int){} ! '
               'videoconvert ! appsink').format(dev_R, width, height, fps)

capL = cv2.VideoCapture(dev_L, cv2.CAP_V4L2)
capR = cv2.VideoCapture(dev_R, cv2.CAP_V4L2)

num = 0


while capL.isOpened() and capR.isOpened():

    successL, imgL = capL.read()
    successR, imgR = capR.read()

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('images/stereoLeft/imageL' + str(num) + '.png', imgL)
        cv2.imwrite('images/stereoRight/imageR' + str(num) + '.png', imgR)
        print("images saved!")
        num += 1

    cv2.imshow('Img L',imgL)
    cv2.imshow('Img R',imgR)
