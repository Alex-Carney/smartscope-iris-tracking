# ASCII only
from typing import Optional, Tuple, Any, Dict
import numpy as np
import cv2

MarkerMeasurement = Tuple[
    Tuple[float, float],  # center_mm (x, y)
    np.ndarray,           # corners_px (4,2)
    Tuple[float, float],  # fov_mm (w, h)
    Tuple[float, float],  # mm_per_px (x, y)
]

# Map valid marker IDs to their semantic role (dynamic vs static).
# IDs not listed here are ignored to suppress spurious detections.
TRACKED_MARKERS: Dict[int, str] = {
    0: "dynamic",
    2: "static",
}

def _clip(v, lo, hi):
    return max(lo, min(hi, v))

class ArucoTracker:

    def __init__(
        self,
        dict_name: str = "DICT_4X4_50",
        aruco_id: int = 0,
        w_mm: float = 19.05,
        h_mm: float = 19.05,
        frame_size_px: Tuple[int, int] | None = None,
        isotropic_scale: bool = False,
        time_accounting: Any | None = None,

        # ---- NEW speed knobs ----
        roi_first: bool = True,             # try detection in last-known ROI first
        roi_pad_px: int = 60,               # pad around last bbox (each side)
        roi_fail_reset: int = 2,            # if ROI misses this many times, go full frame
        full_downscale: float = 0.5,        # full-frame fallback scale (0.5 ~ 4x fewer px)
        tune_params_fast: bool = True,      # tighten DetectorParameters
    ):
        self.aruco_id = aruco_id
        self.w_mm = float(w_mm)
        self.h_mm = float(h_mm)
        self.frame_w_px, self.frame_h_px = (frame_size_px or (0, 0))
        self.isotropic_scale = bool(isotropic_scale)
        self.timer = time_accounting

        # Detector + (optional) faster parameter tuning
        dictionary = getattr(cv2.aruco, dict_name)
        self.dict = cv2.aruco.getPredefinedDictionary(dictionary)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE

        if tune_params_fast:
            # Keep detector from sweeping too many scales/windows:
            # (these rates are relative to min(image dim))
            params.minMarkerPerimeterRate = 0.02    # ignore too-small contours
            params.maxMarkerPerimeterRate = 0.50    # ignore huge “markers”
            # Reduce the adaptive threshold workloads
            params.adaptiveThreshWinSizeMin = 5
            params.adaptiveThreshWinSizeMax = 23
            params.adaptiveThreshWinSizeStep = 6
            params.adaptiveThreshConstant = 7
            # Avoid looking near the image border
            params.minDistanceToBorder = 3
            # For cleanliness
            params.minCornerDistanceRate = 0.03
            params.polygonalApproxAccuracyRate = 0.03

        self.detector = cv2.aruco.ArucoDetector(self.dict, params)

        # ROI tracking state
        self._have_roi = False
        self._roi = (0, 0, 0, 0)   # x0,y0,x1,y1 inclusive
        self._roi_miss = 0
        self.roi_first = roi_first
        self.roi_pad_px = int(roi_pad_px)
        self.roi_fail_reset = int(roi_fail_reset)
        self.full_downscale = float(full_downscale)

    # ---- utility for timing marks ----
    @staticmethod
    def _mark(timer: Any | None, name: str) -> None:
        if timer is not None:
            timer.mark(name)

    # ---- build/update ROI from corners ----
    def _bbox_from_corners(self, c4x2: np.ndarray) -> Tuple[int, int, int, int]:
        x0 = int(np.floor(np.min(c4x2[:, 0])))
        y0 = int(np.floor(np.min(c4x2[:, 1])))
        x1 = int(np.ceil (np.max(c4x2[:, 0])))
        y1 = int(np.ceil (np.max(c4x2[:, 1])))
        if self.roi_pad_px > 0:
            x0 -= self.roi_pad_px; y0 -= self.roi_pad_px
            x1 += self.roi_pad_px; y1 += self.roi_pad_px
        W = self.frame_w_px if self.frame_w_px > 0 else None
        H = self.frame_h_px if self.frame_h_px > 0 else None
        if W is None or H is None:
            # infer from last image we saw — safe clipping later
            return x0, y0, x1, y1
        x0 = _clip(x0, 0, W-1); y0 = _clip(y0, 0, H-1)
        x1 = _clip(x1, 0, W-1); y1 = _clip(y1, 0, H-1)
        return x0, y0, x1, y1

    def _update_roi(self, c4x2: np.ndarray, frame_shape: Tuple[int, int]) -> None:
        H, W = frame_shape[:2]
        x0, y0, x1, y1 = self._bbox_from_corners(c4x2)
        x0 = _clip(x0, 0, W-1); y0 = _clip(y0, 0, H-1)
        x1 = _clip(x1, 0, W-1); y1 = _clip(y1, 0, H-1)
        self._roi = (x0, y0, x1, y1)
        self._have_roi = True
        self._roi_miss = 0

    # ---- ROI-first detect ----
    def _detect_in_roi(self, gray: np.ndarray):
        x0, y0, x1, y1 = self._roi
        if x1 <= x0 or y1 <= y0:
            return None
        roi = gray[y0:y1+1, x0:x1+1]
        corners, ids, _ = self.detector.detectMarkers(roi)
        if corners is None or len(corners) == 0 or ids is None:
            return None
        # translate corners back to image coords
        cc = [c.astype(np.float32) + np.array([[[x0, y0]]], dtype=np.float32) for c in corners]
        return cc, ids

    # ---- full-frame detect with optional downscale ----
    def _detect_full(self, gray: np.ndarray):
        if self.full_downscale < 1.0:
            scale = self.full_downscale
            small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            corners, ids, _ = self.detector.detectMarkers(small)
            if corners is None or len(corners) == 0 or ids is None:
                return None
            # scale corners back
            s = 1.0 / scale
            cc = [c.astype(np.float32) * s for c in corners]
            return cc, ids
        else:
            return self.detector.detectMarkers(gray)

    # ---- your existing safe subpix, now reused ----
    @staticmethod
    def _safe_refine_subpix(gray: np.ndarray, c_in: np.ndarray, timer: Any | None = None) -> np.ndarray:
        try:
            ArucoTracker._mark(timer, "aruco:roi")
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

            ArucoTracker._mark(timer, "aruco:blur")
            roi_blur = cv2.GaussianBlur(roi, (11, 11), 0.6)

            ArucoTracker._mark(timer, "aruco:cornerSubPix")
            refined = cv2.cornerSubPix(
                roi_blur, c_loc,
                (legal_win, legal_win), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 7500, 1e-9)
            )
            refined[:, :, 0] += x0
            refined[:, :, 1] += y0
            return refined
        except Exception:
            return c_in

    def _corners_to_mm(
        self,
        corners_px: np.ndarray,
        frame_shape: Tuple[int, int],
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
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
        x_mm = float(ctr_px[0] * mm_per_px_x)
        y_mm = float(ctr_px[1] * mm_per_px_y)

        H, W = frame_shape
        if self.frame_w_px > 0 and self.frame_h_px > 0:
            fov_w_mm = float(self.frame_w_px * mm_per_px_x)
            fov_h_mm = float(self.frame_h_px * mm_per_px_y)
        else:
            fov_w_mm = float(W * mm_per_px_x)
            fov_h_mm = float(H * mm_per_px_y)

        return (x_mm, y_mm), (fov_w_mm, fov_h_mm), (mm_per_px_x, mm_per_px_y)

    def detect_mm(
        self,
        gray: np.ndarray,
        undistort_points_fn=None,
        timer: Any | None = None,
    ) -> Optional[Tuple[Tuple[float, float], np.ndarray, Tuple[float, float]]]:

        t = timer or self.timer
        self._mark(t, "aruco:start")

        H, W = gray.shape[:2]

        # ---- 1) ROI-first (if we have one) ----
        corners = ids = None
        if self.roi_first and self._have_roi and self._roi_miss < self.roi_fail_reset:
            self._mark(t, "aruco:detectROI")
            roi_ret = self._detect_in_roi(gray)
            if roi_ret is not None:
                corners, ids = roi_ret
            else:
                self._roi_miss += 1

        # ---- 2) full-frame fallback (optional downscale) ----
        if corners is None:
            self._mark(t, "aruco:detectFull")
            full_ret = self._detect_full(gray)
            if full_ret is None:
                return None
            corners, ids = full_ret
            self._roi_miss = 0  # reset since full-frame succeeded

        # ---- pick marker (target id or first) ----
        if ids is None or len(corners) == 0:
            return None

        allowed = []
        for idx, marker_id_arr in enumerate(ids):
            marker_id = int(marker_id_arr[0])
            if marker_id not in TRACKED_MARKERS:
                continue
            allowed.append((idx, marker_id))

        if not allowed:
            return None

        if self.aruco_id is not None:
            target_idx = next((idx for idx, mid in allowed if mid == self.aruco_id), None)
            if target_idx is None:
                return None
        else:
            target_idx = allowed[0][0]

        self._mark(t, "aruco:select_id")

        c = corners[target_idx].astype(np.float32).reshape(1, 4, 2)

        # Optional undistort
        if undistort_points_fn is not None:
            c = undistort_points_fn(c)
        self._mark(t, "aruco:undistort")

        # Subpixel refinement
        c_ref = self._safe_refine_subpix(gray, c, timer=t)
        corners_px = c_ref.reshape(4, 2).astype(np.float32)

        # Update ROI for next frame
        self._update_roi(corners_px, (H, W))

        # mm conversion
        self._mark(t, "aruco:mm")
        mm_info = self._corners_to_mm(corners_px, (H, W))
        if mm_info is None:
            return None
        center_mm, fov_mm, _scale = mm_info
        return center_mm, corners_px, fov_mm

    def detect_multiple_mm(
        self,
        gray: np.ndarray,
        undistort_points_fn=None,
        timer: Any | None = None,
    ) -> Dict[int, MarkerMeasurement]:
        """
        Detect all visible markers in `gray` and return per-ID millimeter data.
        """
        t = timer or self.timer
        self._mark(t, "aruco_multi:start")

        # Full-frame detection (multi-marker path does not reuse ROI heuristics yet)
        self._mark(t, "aruco_multi:detectFull")
        full_ret = self._detect_full(gray)
        if full_ret is None:
            return {}
        corners_list, ids = full_ret

        detections: Dict[int, MarkerMeasurement] = {}
        H, W = gray.shape[:2]

        for raw_corners, marker_id_arr in zip(corners_list, ids):
            marker_id = int(marker_id_arr[0])
            if marker_id not in TRACKED_MARKERS:
                continue
            c = raw_corners.astype(np.float32).reshape(1, 4, 2)

            self._mark(t, "aruco_multi:prep")
            if undistort_points_fn is not None:
                c = undistort_points_fn(c)
                self._mark(t, "aruco_multi:undistort")

            refined = self._safe_refine_subpix(gray, c, timer=t)
            self._mark(t, "aruco_multi:refine")
            corners_px = refined.reshape(4, 2).astype(np.float32)

            mm_info = self._corners_to_mm(corners_px, (H, W))
            self._mark(t, "aruco_multi:mm")
            if mm_info is None:
                continue
            center_mm, fov_mm, mm_per_px = mm_info

            detections[marker_id] = (center_mm, corners_px, fov_mm, mm_per_px)

            if marker_id == self.aruco_id:
                self._update_roi(corners_px, (H, W))

        self._mark(t, "aruco_multi:end")
        return detections
