from typing import Optional

import cv2
import numpy as np


Point = tuple[float, float]
ROI = tuple[int, int, int, int]
TrackingResult = tuple[Optional[Point], float]


class VisionTracker:
    """カメラ画像からビーム・試料・視野の位置変化を追跡する。"""

    MIN_BEAM_INTENSITY = 30.0

    def __init__(
        self,
        auto_ratio: float = 0.90,
        min_area: float = 3.0,
    ) -> None:
        # Beam detection
        self.auto_ratio = auto_ratio
        self.min_area = min_area

        # V1: Template matching
        self.template: Optional[np.ndarray] = None
        self.target_offset: Optional[Point] = None

        # V2: Phase correlation
        self.ref_gray: Optional[np.ndarray] = None
        self._hann_window: Optional[np.ndarray] = None

    # ==========================================================================
    # Common
    # ==========================================================================

    def detect_beam(
        self,
        gray_img: np.ndarray,
        thresh: Optional[float] = None,
    ) -> Optional[Point]:
        """
        光スポット中心 (x, y) を検出する。

        Returns
        -------
        Point | None
            光スポット中心。検出できなければ None。
        """
        max_val = float(np.max(gray_img))

        if max_val < self.MIN_BEAM_INTENSITY:
            return None

        threshold = self._get_beam_threshold(max_val, thresh)
        mask = self._create_beam_mask(gray_img, threshold)
        contour = self._find_beam_contour(mask)

        if contour is None:
            return None

        return self._calculate_centroid(contour)

    def _get_beam_threshold(
        self,
        max_val: float,
        thresh: Optional[float],
    ) -> float:
        """ビーム検出用の二値化閾値を決定する。"""
        if thresh is not None:
            return float(thresh)

        threshold = max_val * self.auto_ratio

        if threshold >= max_val:
            threshold = max_val - 1.0

        return max(0.0, threshold)

    @staticmethod
    def _create_beam_mask(
        gray_img: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        """画像を二値化する。"""
        _, mask = cv2.threshold(
            gray_img,
            threshold,
            255,
            cv2.THRESH_BINARY,
        )
        return mask

    def _find_beam_contour(
        self,
        mask: np.ndarray,
    ) -> Optional[np.ndarray]:
        """最大面積のビーム輪郭を取得する。"""
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(contour) < self.min_area:
            return None

        return contour

    @staticmethod
    def _calculate_centroid(
        contour: np.ndarray,
    ) -> Optional[Point]:
        """輪郭の重心を計算する。"""
        moments = cv2.moments(contour)

        if moments["m00"] == 0:
            return None

        return (
            moments["m10"] / moments["m00"],
            moments["m01"] / moments["m00"],
        )

    @staticmethod
    def _calculate_offset(
        beam_pos: Point,
        sample_pos: Point,
    ) -> Point:
        """ビームから試料への相対位置を計算する。"""
        return (
            sample_pos[0] - beam_pos[0],
            sample_pos[1] - beam_pos[1],
        )

    # ==========================================================================
    # V1: Template matching
    # ==========================================================================

    def set_template(
        self,
        gray_img: np.ndarray,
        roi: ROI,
    ) -> None:
        """追従対象の矩形 (x, y, w, h) をテンプレートとして登録する。"""
        x, y, width, height = roi

        self.template = gray_img[
            y:y + height,
            x:x + width,
        ].copy()

    def track_v1_template(
        self,
        gray_img: np.ndarray,
    ) -> TrackingResult:
        """
        テンプレートマッチングで試料位置を検出する。

        Returns
        -------
        ((x, y), score)
            試料中心位置とマッチングスコア。
            テンプレート未登録時は (None, 0.0)。
        """
        if self.template is None:
            return None, 0.0

        result = cv2.matchTemplate(
            gray_img,
            self.template,
            cv2.TM_CCOEFF_NORMED,
        )

        _, score, _, location = cv2.minMaxLoc(result)

        template_height, template_width = self.template.shape[:2]

        position = (
            location[0] + template_width / 2.0,
            location[1] + template_height / 2.0,
        )

        return position, score

    def lock_v1_target(
        self,
        beam_pos: Point,
        sample_pos: Point,
    ) -> None:
        """現在のビームと試料の相対位置を基準として保存する。"""
        self.target_offset = self._calculate_offset(
            beam_pos,
            sample_pos,
        )

    def calculate_v1_drift(
        self,
        beam_pos: Point,
        sample_pos: Point,
    ) -> Optional[Point]:
        """基準位置からの相対位置のずれ量 (dx, dy) [px] を計算する。"""
        if self.target_offset is None:
            return None

        current_offset = self._calculate_offset(
            beam_pos,
            sample_pos,
        )

        return (
            current_offset[0] - self.target_offset[0],
            current_offset[1] - self.target_offset[1],
        )

    # ==========================================================================
    # V2: Phase correlation
    # ==========================================================================

    def set_ref_frame(
        self,
        gray_img: np.ndarray,
    ) -> None:
        """位相相関の基準画像を登録する。"""
        self.ref_gray = gray_img.copy()

        height, width = gray_img.shape
        self._hann_window = cv2.createHanningWindow(
            (width, height),
            cv2.CV_32F,
        )

    def track_v2_background(
        self,
        gray_img: np.ndarray,
        beam_pos: Optional[Point] = None,
        mask_r: int = 35,
    ) -> TrackingResult:
        """
        位相限定相関により視野全体のシフト量を検出する。

        Returns
        -------
        ((dx, dy), response)
            基準画像から現在画像へのシフト量 [px] と
            位相相関の response。
        """
        if self.ref_gray is None:
            return None, 0.0

        if self._hann_window is None:
            return None, 0.0

        ref_proc = self.ref_gray.copy()
        cur_proc = gray_img.copy()

        if beam_pos is not None:
            self._mask_beam(
                ref_proc,
                cur_proc,
                beam_pos,
                mask_r,
            )

        (dx, dy), response = cv2.phaseCorrelate(
            ref_proc.astype(np.float32),
            cur_proc.astype(np.float32),
            self._hann_window,
        )

        return (dx, dy), response

    def _mask_beam(
        self,
        ref_img: np.ndarray,
        cur_img: np.ndarray,
        beam_pos: Point,
        radius: int,
    ) -> None:
        """ビーム領域をマスクする。"""
        x = int(round(beam_pos[0]))
        y = int(round(beam_pos[1]))

        ref_value = int(np.median(ref_img))
        cur_value = int(np.median(cur_img))

        cv2.circle(
            ref_img,
            (x, y),
            radius,
            ref_value,
            -1,
        )

        cv2.circle(
            cur_img,
            (x, y),
            radius,
            cur_value,
            -1,
        )
