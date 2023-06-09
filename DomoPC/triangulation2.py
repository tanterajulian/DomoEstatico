# import numpy as np
# import cv2

# def find_depth(right_point, left_point, frame_right, frame_left, baseline, f, alpha):
#     # Convert baseline from mm to cm
#     baseline_cm = baseline / 10

#     # Convert focal length from mm to cm
#     f_cm = f / 10

#     # Detect keypoints and extract descriptors using SIFT
#     sift = cv2.SIFT_create()
#     kp1, des1 = sift.detectAndCompute(frame_right, None)
#     kp2, des2 = sift.detectAndCompute(frame_left, None)

#     # Match keypoints using FLANN matcher
#     FLANN_INDEX_KDTREE = 1
#     index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
#     search_params = dict()
#     flann = cv2.FlannBasedMatcher(index_params, search_params)
#     matches = flann.knnMatch(des1, des2, k=2)

#     # Apply Lowe's ratio test to filter matches
#     good_matches = []
#     for m, n in matches:
#         if m.distance < 0.7 * n.distance:
#             good_matches.append(m)

#     # Extract matched keypoints
#     points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#     points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

#     # Find essential matrix using RANSAC algorithm
#     E, mask = cv2.findEssentialMat(points1, points2, f_cm, (0, 0), cv2.RANSAC, 0.999, 1.0)

#     # Recover relative camera pose from essential matrix
#     _, R, t, mask = cv2.recoverPose(E, points1, points2, focal=f_cm, pp=(0, 0))

#     # Calculate disparity in pixels
#     dist = np.linalg.norm(t)
#     disparity = abs(right_point[0] - left_point[0])
    
#     pixel_per_cm = frame_right.shape[1] / 2.54
    
#     # Calculate depth in cm
#     zDepth = (f_cm * baseline_cm * pixel_per_cm) / (disparity + alpha)

#     return zDepth


# triangulacion pìxeles a cm

import numpy as np
import cv2

def find_depth(right_point, left_point, frame_right, frame_left, baseline, f, alpha):
    # Convert baseline from mm to cm
    baseline_cm = baseline / 10

    # Convert focal length from mm to cm
    f_cm = f / 10

    # Calculate disparity in pixels
    disparity = abs(right_point[0] - left_point[0])

    # Convert image resolution to cm
    pixel_per_cm = frame_right.shape[1] / 2.54

    # Calculate depth in cm
    zDepth = (f_cm * baseline_cm * pixel_per_cm) / (disparity + alpha)

    return zDepth

# triangulacion pìxeles a mm 

# import numpy as np
# import cv2

# def find_depth(right_point, left_point, frame_right, frame_left, baseline, f, alpha):
#     # Convert focal length from mm to pixels
#     f_px = f / (frame_right.shape[1] / 2)

#     # Calculate disparity
#     disparity = abs(right_point[0] - left_point[0])

#     # Calculate depth
#     zDepth = (f_px * baseline) / (disparity + alpha)

#     return zDepth

# # Example usage
# # Load left and right frames
# frame_right = cv2.imread('frame_right.png')
# frame_left = cv2.imread('frame_left.png')

# # Define baseline, focal length, and alpha value
# baseline = 0.1 # meters
# f_mm = 10 # millimeters
# alpha = 0 # no additional parameter added to the disparity value

# # Define corresponding points in the right and left frames
# right_point = (100, 200)
# left_point = (70, 200)

# # Calculate depth
# depth = find_depth(right_point, left_point, frame_right, frame_left, baseline, f_mm, alpha)

# print('Depth:', depth, 'meters')
