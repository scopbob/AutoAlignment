import cv2
from vision_tracker import VisionTracker


def main():
    cap = cv2.VideoCapture(0)  # カメラデバイス (ファイルなら 'path/to/video.mp4')
    tracker = VisionTracker(auto_ratio=0.90)

    window_name = "Tracker (Press 'r': Lock, 'q': Quit)"
    cv2.namedWindow(window_name)

    print("=== 操作方法 ===")
    print(" [r] : 追従したい結晶をドラッグ選択してロック")
    print(" [q] : 終了")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. 各座標の検出
        beam_pos = tracker.detect_beam(gray)
        sample_pos, score = tracker.track_sample(gray)

        # 2. ビーム座標の描画
        if beam_pos is not None:
            bx, by = beam_pos
            cv2.circle(frame, (int(bx), int(by)), 4, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"Beam: ({bx:.2f}, {by:.2f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        # 3. 試料座標とドリフトの計算・描画
        if sample_pos is not None and score > 0.6:
            sx, sy = sample_pos
            th, tw = tracker.template.shape[:2]
            tl = (int(sx - tw / 2), int(sy - th / 2))
            cv2.rectangle(frame, tl, (tl[0] + tw, tl[1] + th), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Sample: ({sx:.2f}, {sy:.2f})",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            if beam_pos is not None:
                drift = tracker.calculate_drift(beam_pos, sample_pos)
                if drift is not None:
                    dx, dy = drift
                    cv2.putText(
                        frame,
                        f"Drift: dX={dx:+.2f}px, dY={dy:+.2f}px",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2,
                    )
                    print(
                        f"Beam: ({bx:6.2f}, {by:6.2f}) | Sample: ({sx:6.2f}, {sy:6.2f}) | Drift: dX={dx:+5.2f}, dY={dy:+5.2f}"
                    )

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            print("ドラッグして結晶を囲み、Space または Enter を押してください...")
            rx, ry, rw, rh = cv2.selectROI(window_name, frame, False)
            if rw > 10 and rh > 10:
                tracker.set_template(gray, (rx, ry, rw, rh))
                init_sample, _ = tracker.track_sample(gray)
                init_beam = tracker.detect_beam(gray)
                if init_beam is not None and init_sample is not None:
                    tracker.lock_target(init_beam, init_sample)
                    print("--> ターゲットロック完了")
                else:
                    print("--> 警告: ビームまたは試料が認識できませんでした")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
