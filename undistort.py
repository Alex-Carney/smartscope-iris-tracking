from typing import Tuple
import numpy as np
import cv2

class Undistorter:
    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, image_size: Tuple[int, int]):
        self.K = camera_matrix
        self.D = dist_coeffs
        self.size = image_size  # (w, h)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(self.K, self.D, None, self.K, self.size, cv2.CV_16SC2)

    def remap(self, frame_bgr: np.ndarray) -> np.ndarray:
        return cv2.remap(frame_bgr, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)

    def undistort_points(self, corners: np.ndarray) -> np.ndarray:
        # corners shape (1,4,2) float32
        und = cv2.undistortPoints(corners, self.K, self.D, P=self.K)
        return und.reshape(1, 4, 2)