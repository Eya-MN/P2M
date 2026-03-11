import time
from collections import Counter, deque

import cv2
from ultralytics import YOLO

MODEL_PATH = r"runs\classify\sign_classification_ultra_fast2\weights\best.pt"
CAMERA_INDEX = 0

# Paramètres simples
IMG_SIZE = 224
CONF_THRESHOLD = 0.70  # Reasonable threshold
SMOOTHING_WINDOW = 5
PREDICT_EVERY_N_FRAMES = 3

# Simple center crop
USE_SIMPLE_CROP = True
CROP_PERCENTAGE = 0.6

def majority_vote(items: deque[str]) -> str | None:
    if not items:
        return None
    return Counter(items).most_common(1)[0][0]

def simple_center_crop(frame):
    """Simple center crop"""
    h, w = frame.shape[:2]
    
    crop_h = int(h * CROP_PERCENTAGE)
    crop_w = int(w * CROP_PERCENTAGE)
    
    start_x = (w - crop_w) // 2
    start_y = (h - crop_h) // 2
    
    end_x = start_x + crop_w
    end_y = start_y + crop_h
    
    roi = frame[start_y:end_y, start_x:end_x]
    return roi, (start_x, start_y, end_x, end_y)

def main() -> None:
    print("🎥 WEBCAM RECONNAISSANCE SIGNES")
    print("=" * 40)
    print("💡 Instructions:")
    print("   - Placez vos mains au centre")
    print("   - Cadre vert = zone de détection")
    print("   - 'q' = quitter, 'c' = toggle crop")
    print("=" * 40)
    
    # Charger le modèle
    try:
        model = YOLO(MODEL_PATH)
        print("✅ Modèle chargé")
    except Exception as e:
        print(f"❌ Erreur modèle: {e}")
        return
    
    # Webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ Webcam non disponible")
        return
    
    print("✅ Webcam démarrée")
    
    recent_preds: deque[str] = deque(maxlen=SMOOTHING_WINDOW)
    last_label = "unknown"
    last_conf = 0.0
    
    frame_idx = 0
    t0 = time.time()
    fps = 0.0
    cropping_enabled = USE_SIMPLE_CROP
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        frame_idx += 1
        
        # Simple crop
        if cropping_enabled:
            roi, crop_box = simple_center_crop(frame)
        else:
            roi = frame
            crop_box = None
        
        # Classification
        if frame_idx % PREDICT_EVERY_N_FRAMES == 0:
            try:
                result = model.predict(roi, imgsz=IMG_SIZE, verbose=False)[0]
                top1_idx = int(result.probs.top1)
                conf = float(result.probs.top1conf)
                label = result.names[top1_idx]
                
                if conf >= CONF_THRESHOLD:
                    last_label = label
                    last_conf = conf
                else:
                    last_label = "unknown"
                    last_conf = conf
                
                recent_preds.append(last_label)
                
            except Exception as e:
                print(f"⚠️ Erreur: {e}")
                last_label = "error"
                last_conf = 0.0
        
        # Lissage
        if len(recent_preds) >= 3:
            smoothed = majority_vote(recent_preds)
            if smoothed:
                last_label = smoothed
        
        # FPS
        dt = time.time() - t0
        if dt >= 0.5:
            fps = frame_idx / dt
        
        # Interface
        text1 = f"Signe: {last_label}"
        text2 = f"Confiance: {last_conf*100:.1f}% | FPS: {fps:.1f}"
        text3 = "q=quitter | c=toggle crop"
        text4 = f"Mode: {'CENTRÉ' if cropping_enabled else 'PLEIN'}"
        
        # Fond texte
        cv2.rectangle(frame, (10, 10), (500, 140), (0, 0, 0), thickness=-1)
        cv2.putText(frame, text1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, text2, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(frame, text3, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        cv2.putText(frame, text4, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        
        # Dessiner la zone de détection
        if crop_box is not None:
            x1, y1, x2, y2 = crop_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            # Marquer le centre
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
        
        cv2.imshow("Reconnaissance Signes", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            cropping_enabled = not cropping_enabled
            print(f"🔄 Mode: {'CENTRÉ' if cropping_enabled else 'PLEIN'}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Webcam fermée")

if __name__ == "__main__":
    main()
