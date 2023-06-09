import numpy as np
import cv2
import glob

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def calibrate(image_dir, prefix, image_format, square_size, width, height):
    """Apply camera calibration operation for images in the given directory."""
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    objp = objp * square_size

    # Arrays to store object points and image points from all the images
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    if image_dir[-1:] == '/':
        image_dir = image_dir[:-1]

    images = glob.glob(image_dir + '/' + prefix + '*.' + image_format)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # If found, add object points and image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)
            cv2.imshow("Chessboard", img)
            cv2.waitKey(500)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    return [ret, mtx, dist, rvecs, tvecs]


def save_coefficients(mtx, dist, path):
    """Save the camera matrix and the distortion coefficients to the given path/file."""
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("K", mtx)
    cv_file.write("D", dist)
    # Note: You *release* the FileStorage object, you don't close() it
    cv_file.release()


def load_coefficients(path):
    """Load camera matrix and distortion coefficients."""
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()
    cv_file.release()
    return camera_matrix, dist_matrix


if __name__ == '__main__':
    image_dir = "images/stereoRight"
    image_format = "png"
    prefix = "imageR"
    square_size = 0.025
    width = 9
    height = 6
    save_file = "camarader.yml"

    ret, mtx, dist, rvecs, tvecs = calibrate(image_dir, prefix, image_format, square_size, width, height)
    save_coefficients(mtx, dist, save_file)
    print("Finalizo la calibracion de la camara derecha. RMS:", ret)

    # El valor RMS (Root Mean Square) representa la raíz cuadrada de la media de los cuadrados de los errores de la calibración de la cámara

    image_dir = "images/stereoLeft"
    image_format = "png"
    prefix = "imageL"
    square_size = 0.025
    width = 9
    height = 6
    save_file = "camaraizq.yml"

    ret, mtx, dist, rvecs, tvecs = calibrate(image_dir, prefix, image_format, square_size, width, height)
    save_coefficients(mtx, dist, save_file)
    print("Finalizo la calibracion de la camara izquierda. RMS:", ret)