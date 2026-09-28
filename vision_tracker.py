from typing import Optional, Tuple
import cv2
import numpy as np


class VisionTracker:

    def __init__(self, auto_ratio: float = 0.90, min_area: float = 3.0):
        self.auto_ratio = auto_ratio
        self.min_area = min_area
        self.template: Optional[np.ndarray] = None
        self.target_offset: Optional[Tuple[float, float]] = None

    def detect_beam(
        self, gray_img: np.ndarray, thresh: Optional[float] = None
    ) -> Optional[Tuple[float, float]]:
        """光スポット中心 (x, y) をサブピクセル重心で検出。"""
        max_val = np.max(gray_img)
        if max_val < 30:
            return None

        th = float(thresh) if thresh is not None else max_val * self.auto_ratio
        if th >= max_val:
            th = max(0.0, max_val - 1.0)

        _, mask = cv2.threshold(gray_img, th, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not cnts:
            return None

        max_cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(max_cnt) < self.min_area:
            return None

        M = cv2.moments(max_cnt)
        if M["m00"] != 0:
            return (M["m10"] / M["m00"], M["m01"] / M["m00"])
        return None

    def set_template(self, gray_img: np.ndarray, roi: Tuple[int, int, int, int]):
        """追従対象の矩形 (x, y, w, h) からテンプレートを登録。"""
        rx, ry, rw, rh = roi
        self.template = gray_img[ry : ry + rh, rx : rx + rw].copy()

    def track_sample(
        self, gray_img: np.ndarray
    ) -> Tuple[Optional[Tuple[float, float]], float]:
        """登録したテンプレートの位置 (x, y) とスコアを算出。"""
        if self.template is None:
            return None, 0.0

        res = cv2.matchTemplate(gray_img, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        th, tw = self.template.shape[:2]
        cx = max_loc[0] + tw / 2.0
        cy = max_loc[1] + th / 2.0
        return (cx, cy), max_val

    def lock_target(
        self, beam_pos: Tuple[float, float], sample_pos: Tuple[float, float]
    ):
        """初期のビームと試料の相対ベクトルをロック。"""
        self.target_offset = (
            sample_pos[0] - beam_pos[0],
            sample_pos[1] - beam_pos[1],
        )

    def calculate_drift(
        self, beam_pos: Tuple[float, float], sample_pos: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """現在の位置関係から、ロック時からのズレ量 (drift_x, drift_y) [px] を計算。"""
        if self.target_offset is None:
            return None
        current_dx = sample_pos[0] - beam_pos[0]
        current_dy = sample_pos[1] - beam_pos[1]
        drift_x = current_dx - self.target_offset[0]
        drift_y = current_dy - self.target_offset[1]
        return (drift_x, drift_y)
