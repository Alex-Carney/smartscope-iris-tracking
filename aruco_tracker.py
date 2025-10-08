# ASCII only
from typing import Optional, Tuple, Any
import numpy as np
import cv2

class ArucoTracker:
    CENTER_STAGE_X = 50
    CENTER_STAGE_Y = 50

    def __init__(
        self,
        dict_name: str = "DICT_4X4_50",
        aruco_id: int = 0,
        w_mm: float = 19.05,
        h_mm: float = 19.05,
        frame_size_px: Tuple[int, int] | None = None,  # (W,H)
        isotropic_scale: bool = False,                 # use same mm/px for X and Y
        time_accounting: Any | None = None,            # optional TimeAccounting
    ):
        self.aruco_id = aruco_id
        self.w_mm = float(w_mm)
        self.h_mm = float(h_mm)
        self.frame_w_px, self.frame_h_px = (frame_size_px or (0, 0))
        self.isotropic_scale = bool(isotropic_scale)
        self.timer = time_accounting  # stored default timer (can override per call)

        dictionary = getattr(cv2.aruco, dict_name)
        self.dict = cv2.aruco.getPredefinedDictionary(dictionary)
        params = cv2.aruco.DetectorParameters()
        # We do manual refinement, so disable built-in corner refinement here
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
        self.detector = cv2.aruco.ArucoDetector(self.dict, params)

    @staticmethod
    def _safe_refine_subpix(
        gray: np.ndarray,
        c_in: np.ndarray,
        timer: Any | None = None,
    ) -> np.ndarray:
        """
        Returns refined corners or falls back to c_in on any error.
        Emits fine-grained timing marks if `timer` is provided.
        """
        try:
            c = c_in
            # ----- ROI build -----
            if timer is not None:
                timer.mark("aruco:roi")
            xmin = float(np.min(c[0, :, 0])); xmax = float(np.max(c[0, :, 0]))
            ymin = float(np.min(c[0, :, 1])); ymax = float(np.max(c[0, :, 1]))

            DESIRED_WIN = 11
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

            # ----- blur (helps subpix stability) -----
            if timer is not None:
                timer.mark("aruco:blur")
            roi_blur = cv2.GaussianBlur(roi, (9, 9), 0.6)

            # ----- subpixel refinement -----
            if timer is not None:
                timer.mark("aruco:cornerSubPix")
            refined = cv2.cornerSubPix(
                roi_blur, c_loc,
                (legal_win, legal_win), (-1, -1),
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
        undistort_points_fn=None,
        timer: Any | None = None,
    ) -> Optional[Tuple[Tuple[float, float], np.ndarray, Tuple[float, float]]]:
        """
        Returns:
            ((x_mm, y_mm), corners_px(4,2 float32), (fov_w_mm, fov_h_mm))

        Coordinates are in mm with top-left pixel as (0,0).
        """
        t = timer or self.timer
        if t is not None:
            t.mark("aruco:start")

        # ----- detect -----
        corners, ids, _ = self.detector.detectMarkers(gray)
        if t is not None:
            t.mark("aruco:detectMarkers")
        if ids is None or len(corners) == 0:
            return None

        # ----- select target id -----
        if self.aruco_id is not None and ids is not None:
            matches = np.where(ids.flatten() == self.aruco_id)[0]
            if len(matches) == 0:
                return None
            idx = int(matches[0])
        else:
            idx = 0
        if t is not None:
            t.mark("aruco:select_id")

        c = corners[idx].astype(np.float32).reshape(1, 4, 2)

        # ----- optional undistort of corners -----
        if undistort_points_fn is not None:
            c = undistort_points_fn(c)
        if t is not None:
            t.mark("aruco:undistort")

        # ----- subpixel refinement (instrumented internally) -----
        c_ref = self._safe_refine_subpix(gray, c, timer=t)
        corners_px = c_ref.reshape(4, 2).astype(np.float32)

        # ----- mm conversion -----
        if t is not None:
            t.mark("aruco:mm")
        px_w = float(np.linalg.norm(corners_px[1] - corners_px[0]))
        px_h = float(np.linalg.norm(corners_px[2] - corners_px[1]))
        if px_w <= 0.0 or px_h <= 0.0:
            return None

        if self.isotropic_scale:
            s = 0.5 * ((self.w_mm / px_w) + (self.h_mm / px_h))
            mm_per_px_x = mm_per_px_y = s
        else:
            mm_per_px_x = self.w_mm / px_w
            mm_per_px_y = self.h_mm / px_h

        ctr_px = np.mean(corners_px, axis=0)
        x_mm = float(ctr_px[0] * mm_per_px_x) - self.CENTER_STAGE_X
        y_mm = float(ctr_px[1] * mm_per_px_y) - self.CENTER_STAGE_Y

        # FOV in mm
        if self.frame_w_px > 0 and self.frame_h_px > 0:
            fov_w_mm = float(self.frame_w_px * mm_per_px_x)
            fov_h_mm = float(self.frame_h_px * mm_per_px_y)
        else:
            fov_w_mm = float(gray.shape[1] * mm_per_px_x)
            fov_h_mm = float(gray.shape[0] * mm_per_px_y)

        return (x_mm, y_mm), corners_px, (fov_w_mm, fov_h_mm)
