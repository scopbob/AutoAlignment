from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


Point = tuple[float, float]
ROI = tuple[int, int, int, int]


@dataclass(frozen=True)
class TrackingResult:
    """テンプレートマッチング結果。"""

    position: Point
    score: float


class VisionTracker:
    """カメラ画像からビームと試料の位置関係を追跡する。"""

    MIN_BEAM_INTENSITY = 30.0

    def __init__(
        self,
        beam_threshold_ratio: float = 0.90,
        min_beam_area: float = 3.0,
    ) -> None:
        self.beam_threshold_ratio = beam_threshold_ratio
        self.min_beam_area = min_beam_area

        self._template: Optional[np.ndarray] = None
        self._target_offset: Optional[Point] = None

    # ------------------------------------------------------------------
    # Beam detection
    # ------------------------------------------------------------------

    def detect_beam(
        self,
        gray_image: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Optional[Point]:
        """
        光スポットの中心位置を検出する。

        Parameters
        ----------
        gray_image:
            グレースケール画像。
        threshold:
            二値化閾値。Noneの場合は最大輝度から自動決定。

        Returns
        -------
        Point | None
            光スポット中心 (x, y)。検出できなければNone。
        """
        max_intensity = float(np.max(gray_image))

        if max_intensity < self.MIN_BEAM_INTENSITY:
            return None

        threshold = self._calculate_beam_threshold(
            max_intensity,
            threshold,
        )

        mask = self._create_beam_mask(gray_image, threshold)
        contour = self._find_largest_contour(mask)

        if contour is None:
            return None

        return self._calculate_contour_center(contour)

    def _calculate_beam_threshold(
        self,
        max_intensity: float,
        threshold: Optional[float],
    ) -> float:
        """ビーム検出用の二値化閾値を計算する。"""
        if threshold is not None:
            return float(threshold)

        threshold = max_intensity * self.beam_threshold_ratio

        # 全画素が白になるのを防ぐ
        return min(threshold, max_intensity - 1.0)

    @staticmethod
    def _create_beam_mask(
        gray_image: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        """画像を二値化する。"""
        _, mask = cv2.threshold(
            gray_image,
            threshold,
            255,
            cv2.THRESH_BINARY,
        )
        return mask

    def _find_largest_contour(
        self,
        mask: np.ndarray,
    ) -> Optional[np.ndarray]:
        """最大面積の輪郭を取得する。"""
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(contour) < self.min_beam_area:
            return None

        return contour

    @staticmethod
    def _calculate_contour_center(
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

    # ------------------------------------------------------------------
    # Sample tracking
    # ------------------------------------------------------------------

    def set_template(
        self,
        gray_image: np.ndarray,
        roi: ROI,
    ) -> None:
        """
        試料追跡用テンプレートを登録する。

        roi = (x, y, width, height)
        """
        x, y, width, height = roi

        self._template = gray_image[
            y : y + height,
            x : x + width,
        ].copy()

    def track_sample(
        self,
        gray_image: np.ndarray,
    ) -> Optional[TrackingResult]:
        """テンプレートマッチングによって試料位置を検出する。"""
        if self._template is None:
            return None

        result = cv2.matchTemplate(
            gray_image,
            self._template,
            cv2.TM_CCOEFF_NORMED,
        )

        _, score, _, location = cv2.minMaxLoc(result)

        template_height, template_width = self._template.shape[:2]

        center = (
            location[0] + template_width / 2.0,
            location[1] + template_height / 2.0,
        )

        return TrackingResult(
            position=center,
            score=score,
        )

    # ------------------------------------------------------------------
    # Relative position
    # ------------------------------------------------------------------

    def lock_target(
        self,
        beam_position: Point,
        sample_position: Point,
    ) -> None:
        """現在のビームと試料の相対位置を基準位置として保存する。"""
        self._target_offset = self._calculate_offset(
            beam_position,
            sample_position,
        )

    def calculate_drift(
        self,
        beam_position: Point,
        sample_position: Point,
    ) -> Optional[Point]:
        """
        基準位置からの相対位置のずれを計算する。

        Returns
        -------
        Point | None
            (drift_x, drift_y) [px]
        """
        if self._target_offset is None:
            return None

        current_offset = self._calculate_offset(
            beam_position,
            sample_position,
        )

        return (
            current_offset[0] - self._target_offset[0],
            current_offset[1] - self._target_offset[1],
        )

    @staticmethod
    def _calculate_offset(
        beam_position: Point,
        sample_position: Point,
    ) -> Point:
        """ビームから試料への相対位置を計算する。"""
        return (
            sample_position[0] - beam_position[0],
            sample_position[1] - beam_position[1],
        )

    def reset_target(self) -> None:
        """ロックした基準位置を解除する。"""
        self._target_offset = None
