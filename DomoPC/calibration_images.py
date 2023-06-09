import cv2

dev_L = 0
dev_R = 1
width = 1280
height = 720
# Open both cameras
gst_str_L = ('v4l2src device=/dev/video{} ! '
               'video/x-raw, width=(int){}, height=(int){} ! '
               'videoconvert ! appsink').format(dev_L, width, height)
gst_str_R = ('v4l2src device=/dev/video{} ! '
               'video/x-raw, width=(int){}, height=(int){} ! '
               'videoconvert ! appsink').format(dev_R, width, height)

capL = cv2.VideoCapture(gst_str_L, cv2.CAP_GSTREAMER)
capR = cv2.VideoCapture(gst_str_R, cv2.CAP_GSTREAMER)

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
