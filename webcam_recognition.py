import time
from collections import Counter, deque

import cv2
from ultralytics import YOLO

MODEL_PATH = r"runs\classify\sign_classification_improved2\weights\best.pt"
POSE_MODEL_PATH = r"yolov8n-pose.pt"  # pose model (keypoints)
DETECT_MODEL_PATH = r"yolov8n.pt"  # fallback: COCO model (person detection)
CAMERA_INDEX = 0
MAX_CAMERA_INDEX_TO_TRY = 4
CAMERA_DEVICE_NAME = ""  # e.g. "video=Integrated Camera" (DirectShow). Leave empty to use indices.

# Prediction settings
IMG_SIZE = 224
CONF_THRESHOLD = 0.60  # below this, we display 'unknown'
SMOOTHING_WINDOW = 12  # number of recent frames to vote on
PREDICT_EVERY_N_FRAMES = 2  # speed-up (predict 1 frame out of N)

# Cropping pipeline settings
USE_CROPPING = True
DETECT_EVERY_N_FRAMES = 3
PERSON_CLASS_ID = 0
BOX_MARGIN = 0.10  # expand bbox by this fraction
UPPER_BODY_RATIO = 0.75  # keep top part of person box (hands are often in upper area)
MIN_BOX_AREA_RATIO = 0.05  # ignore tiny detections

# Hand crop settings (pose-based)
USE_HAND_CROP = True
HAND_BOX_SCALE = 0.55  # relative to shoulder distance
HAND_BOX_MIN_SIZE = 80  # pixels


def majority_vote(items: deque[str]) -> str | None:
    if not items:
        return None
    return Counter(items).most_common(1)[0][0]


def clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))


def crop_with_person_box(frame, box_xyxy, margin: float, upper_ratio: float):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]

    bw = x2 - x1
    bh = y2 - y1
    mx = int(bw * margin)
    my = int(bh * margin)

    x1 = clamp(x1 - mx, 0, w - 1)
    y1 = clamp(y1 - my, 0, h - 1)
    x2 = clamp(x2 + mx, 1, w)
    y2 = clamp(y2 + my, 1, h)

    # Focus upper part of the person bbox
    y2_upper = y1 + int((y2 - y1) * upper_ratio)
    y2_upper = clamp(y2_upper, y1 + 1, y2)

    roi = frame[y1:y2_upper, x1:x2]
    return roi, (x1, y1, x2, y2_upper)


def crop_with_hand_keypoints(frame, kpts_xy, margin: float):
    """Build a crop box around both hands using pose keypoints.

    kpts_xy: (17, 2) keypoints for a person in COCO format.
    """
    h, w = frame.shape[:2]

    # COCO keypoints indices
    # 5: left_shoulder, 6: right_shoulder, 9: left_wrist, 10: right_wrist
    ls = kpts_xy[5]
    rs = kpts_xy[6]
    lw = kpts_xy[9]
    rw = kpts_xy[10]

    # Basic validity checks (inside frame)
    def valid(p):
        return p is not None and p[0] > 0 and p[1] > 0 and p[0] < w and p[1] < h

    wrists = []
    if valid(lw):
        wrists.append(lw)
    if valid(rw):
        wrists.append(rw)

    if not wrists:
        return None, None

    # Estimate hand box size using shoulder distance (more stable)
    if valid(ls) and valid(rs):
        shoulder_dist = ((ls[0] - rs[0]) ** 2 + (ls[1] - rs[1]) ** 2) ** 0.5
        box_size = int(max(HAND_BOX_MIN_SIZE, shoulder_dist * HAND_BOX_SCALE))
    else:
        box_size = HAND_BOX_MIN_SIZE

    half = box_size // 2

    xs = [p[0] for p in wrists]
    ys = [p[1] for p in wrists]
    cx = int(sum(xs) / len(xs))
    cy = int(sum(ys) / len(ys))

    x1 = clamp(cx - half, 0, w - 1)
    y1 = clamp(cy - half, 0, h - 1)
    x2 = clamp(cx + half, x1 + 1, w)
    y2 = clamp(cy + half, y1 + 1, h)

    # Apply margin
    bw = x2 - x1
    bh = y2 - y1
    mx = int(bw * margin)
    my = int(bh * margin)
    x1 = clamp(x1 - mx, 0, w - 1)
    y1 = clamp(y1 - my, 0, h - 1)
    x2 = clamp(x2 + mx, x1 + 1, w)
    y2 = clamp(y2 + my, y1 + 1, h)

    roi = frame[y1:y2, x1:x2]
    return roi, (x1, y1, x2, y2)


def main() -> None:
    model = YOLO(MODEL_PATH)
    pose = YOLO(POSE_MODEL_PATH) if (USE_CROPPING and USE_HAND_CROP) else None
    detector = YOLO(DETECT_MODEL_PATH) if USE_CROPPING else None

    cap = None
    chosen_index = None
    chosen_backend = None

    backends = [
        ("DSHOW", getattr(cv2, "CAP_DSHOW", None)),
        ("MSMF", getattr(cv2, "CAP_MSMF", None)),
        ("DEFAULT", None),
    ]

    # 0) Optional: open by DirectShow device name (more reliable on some Windows setups)
    if CAMERA_DEVICE_NAME:
        dshow = getattr(cv2, "CAP_DSHOW", None)
        if dshow is not None:
            candidate = cv2.VideoCapture(CAMERA_DEVICE_NAME, dshow)
            if candidate.isOpened():
                cap = candidate
                chosen_index = -1
                chosen_backend = f"DSHOW:{CAMERA_DEVICE_NAME}"
            else:
                candidate.release()

    # 1) Fallback: open by index
    for idx in range(CAMERA_INDEX, MAX_CAMERA_INDEX_TO_TRY + 1):
        if cap is not None:
            break
        for backend_name, backend in backends:
            if backend is None and backend_name != "DEFAULT":
                continue

            candidate = cv2.VideoCapture(idx) if backend is None else cv2.VideoCapture(idx, backend)
            if candidate.isOpened():
                cap = candidate
                chosen_index = idx
                chosen_backend = backend_name
                break
            candidate.release()

        if cap is not None:
            break

    if cap is None or chosen_index is None:
        raise RuntimeError(
            "Cannot open webcam. "
            f"Tried indices {list(range(CAMERA_INDEX, MAX_CAMERA_INDEX_TO_TRY + 1))} "
            "with backends DSHOW/MSMF/DEFAULT."
        )

    recent_preds: deque[str] = deque(maxlen=SMOOTHING_WINDOW)
    last_label = ""
    last_conf = 0.0

    frame_idx = 0
    t0 = time.time()
    fps = 0.0

    # Detection state (we reuse the last bbox for a few frames)
    last_crop_box = None  # (x1, y1, x2, y2)
    last_det_frame = 0
    cropping_enabled = USE_CROPPING

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1

        # 1) Detect hands (pose) then compute crop box (fallback to person)
        roi = frame
        crop_box = None

        if cropping_enabled:
            h, w = frame.shape[:2]

            should_detect = (
                last_crop_box is None
                or (frame_idx - last_det_frame) >= DETECT_EVERY_N_FRAMES
            )

            if should_detect:
                # Try hand crop first (pose)
                got_hand_crop = False
                if pose is not None:
                    pose_res = pose.predict(frame, verbose=False)[0]
                    if pose_res.keypoints is not None and len(pose_res.keypoints) > 0:
                        # select person with largest bbox if available, else first
                        best_i = 0
                        if pose_res.boxes is not None and len(pose_res.boxes) > 0:
                            best_area = 0.0
                            for i, b in enumerate(pose_res.boxes):
                                x1, y1, x2, y2 = b.xyxy[0].tolist()
                                area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
                                if area > best_area:
                                    best_area = area
                                    best_i = i

                        kpts_xy = pose_res.keypoints.xy[best_i].tolist()
                        roi_hand, box_hand = crop_with_hand_keypoints(frame, kpts_xy, margin=BOX_MARGIN)
                        if roi_hand is not None and box_hand is not None and roi_hand.size > 0:
                            roi, crop_box = roi_hand, box_hand
                            last_crop_box = crop_box
                            last_det_frame = frame_idx
                            got_hand_crop = True

                # Fallback to person crop if no hand crop
                if (not got_hand_crop) and detector is not None:
                    det_res = detector.predict(frame, verbose=False, classes=[PERSON_CLASS_ID])[0]
                    best = None
                    best_area = 0.0
                    if det_res.boxes is not None and len(det_res.boxes) > 0:
                        for b in det_res.boxes:
                            x1, y1, x2, y2 = b.xyxy[0].tolist()
                            area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
                            if area > best_area:
                                best_area = area
                                best = (x1, y1, x2, y2)

                    if best is not None and best_area >= (w * h * MIN_BOX_AREA_RATIO):
                        roi, crop_box = crop_with_person_box(
                            frame,
                            best,
                            margin=BOX_MARGIN,
                            upper_ratio=UPPER_BODY_RATIO,
                        )
                        last_crop_box = crop_box
                        last_det_frame = frame_idx
                    else:
                        last_crop_box = None

            if last_crop_box is not None and crop_box is None:
                # reuse last crop box if we skipped detection this frame
                x1, y1, x2, y2 = last_crop_box
                roi = frame[y1:y2, x1:x2]
                crop_box = last_crop_box

        # 2) Classify the ROI
        if frame_idx % PREDICT_EVERY_N_FRAMES == 0:
            # Ultralytics accepts numpy frames directly
            result = model.predict(roi, imgsz=IMG_SIZE, verbose=False)[0]
            top1_idx = int(result.probs.top1)
            conf = float(result.probs.top1conf)
            label = result.names[top1_idx]

            if conf < CONF_THRESHOLD:
                label_to_store = "unknown"
            else:
                label_to_store = label

            recent_preds.append(label_to_store)
            smoothed = majority_vote(recent_preds) or label_to_store

            last_label = smoothed
            last_conf = conf

        # FPS
        dt = time.time() - t0
        if dt >= 0.5:
            fps = frame_idx / dt

        # Overlay
        text1 = f"Gesture: {last_label}"
        text2 = f"Conf: {last_conf*100:.1f}% | FPS: {fps:.1f}"
        text3 = "Keys: q=quit | c=toggle crop"

        if cropping_enabled:
            if USE_HAND_CROP:
                text4 = f"Crop: HANDS (margin={BOX_MARGIN:.2f})"
            else:
                text4 = f"Crop: PERSON (upper={UPPER_BODY_RATIO:.2f}, margin={BOX_MARGIN:.2f})"
        else:
            text4 = "Crop: OFF"

        cv2.rectangle(frame, (10, 10), (620, 130), (0, 0, 0), thickness=-1)
        cv2.putText(frame, text1, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, text2, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(frame, text3, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        cv2.putText(frame, text4, (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        # draw crop box
        if crop_box is not None:
            x1, y1, x2, y2 = crop_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Gesture Recognition (YOLOv8-CLS)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            cropping_enabled = not cropping_enabled
            last_crop_box = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
