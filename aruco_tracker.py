from typing import Optional, Tuple
import numpy as np
import cv2

class ArucoTracker:
    def __init__(self, dict_name: str = "DICT_4X4_50", aruco_id: int = 0, w_mm: float = 19.05, h_mm: float = 19.05):
        self.aruco_id = aruco_id
        self.w_mm = w_mm
        self.h_mm = h_mm
        dictionary = getattr(cv2.aruco, dict_name)
        self.dict = cv2.aruco.getPredefinedDictionary(dictionary)
        params = cv2.aruco.DetectorParameters()
        # We do safe manual refinement, so disable built-in corner refinement here
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
        self.detector = cv2.aruco.ArucoDetector(self.dict, params)

    @staticmethod
    def _safe_refine_subpix(gray: np.ndarray, c_in: np.ndarray) -> np.ndarray:
        try:
            c = c_in
            xmin = float(np.min(c[0, :, 0])); xmax = float(np.max(c[0, :, 0]))
            ymin = float(np.min(c[0, :, 1])); ymax = float(np.max(c[0, :, 1]))
            DESIRED_WIN = 19
            pad = max(12, 2 * DESIRED_WIN + 6)
            x0 = int(max(0, np.floor(xmin) - pad))
            y0 = int(max(0, np.floor(ymin) - pad))
            x1 = int(min(gray.shape[1], np.ceil(xmax) + pad))
            y1 = int(min(gray.shape[0], np.ceil(ymax) + pad))
            roi = gray[y0:y1, x0:x1]
            if roi.size == 0:
                return c_in
            c_loc = c.copy()
            c_loc[0, :, 0] -= x0
            c_loc[0, :, 1] -= y0
            roi_h, roi_w = roi.shape[:2]
            max_win_x = (roi_w - 5) // 2
            max_win_y = (roi_h - 5) // 2
            legal_win = int(max(1, min(DESIRED_WIN, max_win_x, max_win_y)))
            if legal_win < 1:
                return c_in
            roi_blur = cv2.GaussianBlur(roi, (9, 9), 0.6)
            refined = cv2.cornerSubPix(
                roi_blur, c_loc, (legal_win, legal_win), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5000, 1e-8)
            )
            refined[:, :, 0] += x0
            refined[:, :, 1] += y0
            return refined
        except cv2.error:
            return c_in
        except Exception:
            return c_in

    def detect_mm(
        self,
        gray: np.ndarray,
        undistort_points_fn=None
    ) -> Optional[Tuple[Tuple[float, float], np.ndarray]]:
        """
        Detect the ArUco marker, refine corners, and return:
            ((x_mm, y_mm), corners_px)
        where corners_px is shape (4,2) float32 in pixel coordinates.

        Returns None if no marker (or target ID) is found or geometry is degenerate.
        """
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None or len(corners) == 0:
            return None

        # Find target ID if present, else use first marker
        if self.aruco_id is not None and ids is not None:
            matches = np.where(ids.flatten() == self.aruco_id)[0]
            if len(matches) == 0:
                return None
            idx = int(matches[0])
        else:
            idx = 0

        c = corners[idx].astype(np.float32).reshape(1, 4, 2)

        # Optional undistort to pixel coordinates
        if undistort_points_fn is not None:
            c = undistort_points_fn(c)

        # Safe subpixel refinement
        c_ref = self._safe_refine_subpix(gray, c)
        corners_px = c_ref.reshape(4, 2).astype(np.float32)

        # Convert center to mm using per-frame pixel scale
        center_mm = self._px_to_mm_center(corners_px)
        if center_mm is None:
            return None

        return center_mm, corners_px

    def _px_to_mm_center(self, corners: np.ndarray) -> Optional[Tuple[float, float]]:
        c = corners.reshape(4, 2)
        px_w = float(np.linalg.norm(c[1] - c[0]))
        px_h = float(np.linalg.norm(c[2] - c[1]))
        if px_w <= 0.0 or px_h <= 0.0:
            return None
        mm_per_px_x = self.w_mm / px_w
        mm_per_px_y = self.h_mm / px_h
        ctr = np.mean(c, axis=0)
        return float(ctr[0] * mm_per_px_x), float(ctr[1] * mm_per_px_y)
