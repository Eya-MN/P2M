import cv2

START_INDEX = 0
END_INDEX = 10

# Try common backends on Windows
BACKENDS = [
    ("DSHOW", getattr(cv2, "CAP_DSHOW", None)),
    ("MSMF", getattr(cv2, "CAP_MSMF", None)),
    ("DEFAULT", None),
]

# Try a few resolutions (some drivers fail on certain modes)
RESOLUTIONS = [
    (640, 480),
    (1280, 720),
    (1920, 1080),
]


def try_open(idx: int, backend_name: str, backend) -> tuple[bool, str]:
    cap = cv2.VideoCapture(idx) if backend is None else cv2.VideoCapture(idx, backend)
    if not cap.isOpened():
        cap.release()
        return False, "not opened"

    # Try reading a frame
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return False, "opened but cannot read"

    h, w = frame.shape[:2]
    cap.release()
    return True, f"opened+read (frame={w}x{h})"


def main() -> None:
    print("=== Camera probe (OpenCV) ===")
    print(f"Indices: {START_INDEX}..{END_INDEX}")
    print("Backends:")
    for name, b in BACKENDS:
        print(f" - {name}: {b}")
    print("")

    found_any = False

    for idx in range(START_INDEX, END_INDEX + 1):
        for backend_name, backend in BACKENDS:
            if backend is None and backend_name != "DEFAULT":
                continue

            ok, msg = try_open(idx, backend_name, backend)
            status = "OK" if ok else "NO"
            print(f"index={idx:2d} backend={backend_name:7s} -> {status}: {msg}")
            if ok:
                found_any = True

    if not found_any:
        print("\nNo camera could be opened by OpenCV.")
        print("Tips:")
        print("- Close Windows Camera app / Teams / Zoom etc.")
        print("- Check Windows privacy settings: allow desktop apps to access camera.")
        print("- Update webcam driver.")


if __name__ == "__main__":
    main()
