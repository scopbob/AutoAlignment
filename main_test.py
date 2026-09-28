import os
import sys

# 実行ディレクトリのズレによる import エラーを防止
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from vision_tracker import VisionTracker


def create_simulation_frame(sample_x: float, sample_y: float, beam_x: float, beam_y: float):
    """
    テスト用の仮想顕微鏡画像を生成 (640x480)
    - 背景: 一様な輝度 + わずかなノイズ
    - 試料: 移動可能な十字・矩形パターン
    - ビーム: 固定されたガウシアン高輝度スポット
    """
    h, w = 480, 640
    # 背景 (顕微鏡照明を模した明るさ)
    img = np.full((h, w), 120, dtype=np.uint8)
    noise = np.random.normal(0, 3, (h, w)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 1. 試料（特徴的な形状）の描画（暗い影状の結晶）
    sx, sy = int(round(sample_x)), int(round(sample_y))
    cv2.rectangle(img, (sx - 15, sy - 15), (sx + 15, sy + 15), 50, -1)
    cv2.circle(img, (sx + 15, sy - 15), 10, 70, -1)  # 非対称な特徴

    # 2. ビームスポットの描画（高輝度スポット）
    bx, by = int(round(beam_x)), int(round(beam_y))
    # ガウス分布状に光を足す
    y_grid, x_grid = np.ogrid[:h, :w]
    dist_sq = (x_grid - bx) ** 2 + (y_grid - by) ** 2
    beam_intensity = 135 * np.exp(-dist_sq / (2 * (8 ** 2)))

    img = np.clip(img.astype(np.float32) + beam_intensity, 0, 255).astype(np.uint8)
    return img


def main():
    print("=== VisionTracker モジュール結合テスト ===")
    print(" [r] : 追従対象の結晶をドラッグ選択してロック")
    print(" [w/a/s/d] : 結晶を上下左右に動かす (ドリフトのシミュレーション)")
    print(" [q] : 終了")

    tracker = VisionTracker(auto_ratio=0.90)

    # シミュレーション初期座標
    sample_x, sample_y = 300.0, 240.0
    beam_x, beam_y = 360.0, 240.0

    window_name = "Tracker Test (Press 'r' to Lock, 'w/a/s/d' to Drift)"
    cv2.namedWindow(window_name)

    while True:
        # 1. テストフレームの生成
        gray = create_simulation_frame(sample_x, sample_y, beam_x, beam_y)
        frame_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 2. モジュールによる検出
        beam_pos = tracker.detect_beam(gray)
        sample_pos, score = tracker.track_sample(gray)

        # 3. ビーム描画 (赤点)
        if beam_pos is not None:
            bx, by = beam_pos
            cv2.circle(frame_color, (int(bx), int(by)), 4, (0, 0, 255), -1)
            cv2.putText(frame_color, f"Beam: ({bx:.1f}, {by:.1f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 4. 試料描画 (緑枠) & ドリフト計算
        if sample_pos is not None and score > 0.6:
            sx, sy = sample_pos
            th, tw = tracker.template.shape[:2]
            tl = (int(sx - tw / 2), int(sy - th / 2))
            cv2.rectangle(frame_color, tl, (tl[0] + tw, tl[1] + th), (0, 255, 0), 2)
            cv2.putText(frame_color, f"Sample: ({sx:.1f}, {sy:.1f}) Score: {score:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if beam_pos is not None:
                drift = tracker.calculate_drift(beam_pos, sample_pos)
                if drift is not None:
                    dx, dy = drift
                    cv2.putText(frame_color, f"Drift: dX={dx:+.2f}px, dY={dy:+.2f}px", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    print(f"Drift: dX={dx:+6.2f}px, dY={dy:+6.2f}px | Sample Score: {score:.3f}")

        cv2.imshow(window_name, frame_color)
        key = cv2.waitKey(30) & 0xFF

        # 操作
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("ドラッグして黒い結晶パターンを囲み、Space / Enter を押してください...")
            rx, ry, rw, rh = cv2.selectROI(window_name, frame_color, False)
            if rw > 10 and rh > 10:
                tracker.set_template(gray, (rx, ry, rw, rh))
                init_sample, _ = tracker.track_sample(gray)
                init_beam = tracker.detect_beam(gray)
                if init_beam is not None and init_sample is not None:
                    tracker.lock_target(init_beam, init_sample)
                    print("--> ロック完了！ w/a/s/d で動かしてドリフト値を確認してください。")
        # 試料を人工的に動かす (ドリフト発生)
        elif key == ord('a'):  # 左
            sample_x -= 1.0
        elif key == ord('d'):  # 右
            sample_x += 1.0
        elif key == ord('w'):  # 上
            sample_y -= 1.0
        elif key == ord('s'):  # 下
            sample_y += 1.0

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
