import time
from collections import Counter, deque
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = r"runs\classify\sign_classification_ultra_fast2\weights\best.pt"
CAMERA_INDEX = 0

# Paramètres optimisés pour webcam
IMG_SIZE = 256  # Plus grand pour webcam
CONF_THRESHOLD = 0.60  # Plus bas pour webcam
SMOOTHING_WINDOW = 12  # Plus de stabilité
PREDICT_EVERY_N_FRAMES = 4  # Moins fréquent pour stabilité

# Pre-processing pour webcam
USE_PREPROCESSING = True
BRIGHTNESS_FACTOR = 1.2
CONTRAST_FACTOR = 1.1
DENOISE_STRENGTH = 3

def preprocess_webcam_frame(frame):
    """Améliorer la qualité de l'image webcam"""
    
    # Augmenter la luminosité
    frame = cv2.convertScaleAbs(frame, alpha=BRIGHTNESS_FACTOR, beta=10)
    
    # Augmenter le contraste
    frame = cv2.convertScaleAbs(frame, alpha=CONTRAST_FACTOR, beta=0)
    
    # Réduire le bruit
    frame = cv2.fastNlMeansDenoisingColored(frame, None, DENOISE_STRENGTH, DENOISE_STRENGTH, 7, 21)
    
    # Améliorer la netteté
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    frame = cv2.filter2D(frame, -1, kernel)
    
    return frame

def majority_vote(items: deque[str]) -> str | None:
    if not items:
        return None
    return Counter(items).most_common(1)[0][0]

def smart_crop(frame):
    """Crop intelligent pour webcam"""
    h, w = frame.shape[:2]
    
    # Crop légèrement plus grand pour capturer le contexte
    crop_h = int(h * 0.7)
    crop_w = int(w * 0.7)
    
    # Centrer un peu plus bas (zone des mains)
    start_x = (w - crop_w) // 2
    start_y = int(h * 0.3)  # 30% du haut
    
    end_x = start_x + crop_w
    end_y = start_y + crop_h
    
    # S'assurer qu'on reste dans l'image
    start_x = max(0, start_x)
    start_y = max(0, start_y)
    end_x = min(w, end_x)
    end_y = min(h, end_y)
    
    roi = frame[start_y:end_y, start_x:end_x]
    return roi, (start_x, start_y, end_x, end_y)

def main() -> None:
    print("🎥 WEBCAM - RECONNAISSANCE AMÉLIORÉE")
    print("=" * 50)
    print("🚀 AMÉLIORATIONS ACTIVÉES:")
    print("   ✅ Pre-processing image")
    print("   ✅ Crop intelligent")
    print("   ✅ Stabilisation avancée")
    print("   ✅ Threshold optimisé")
    print("=" * 50)
    print("💡 Instructions:")
    print("   - Bon éclairage sur les mains")
    print("   - Fond clair et simple")
    print("   - Placez mains dans le cadre vert")
    print("   - 'q' = quitter, 'p' = toggle pre-processing")
    print("=" * 50)
    
    # Charger le modèle
    try:
        model = YOLO(MODEL_PATH)
        print("✅ Modèle 95.80% chargé")
    except Exception as e:
        print(f"❌ Erreur modèle: {e}")
        return
    
    # Webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ Webcam non disponible")
        return
    
    # Configurer webcam pour meilleure qualité
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✅ Webcam HD démarrée")
    
    recent_preds: deque[str] = deque(maxlen=SMOOTHING_WINDOW)
    last_label = "unknown"
    last_conf = 0.0
    
    frame_idx = 0
    t0 = time.time()
    fps = 0.0
    cropping_enabled = True
    preprocessing_enabled = USE_PREPROCESSING
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        frame_idx += 1
        
        # Pre-processing optionnel
        display_frame = frame.copy()
        if preprocessing_enabled:
            frame = preprocess_webcam_frame(frame)
        
        # Crop intelligent
        if cropping_enabled:
            roi, crop_box = smart_crop(frame)
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
                    print(f"✅ Signe détecté: {label} ({conf:.2f})")
                else:
                    last_label = "unknown"
                    last_conf = conf
                
                recent_preds.append(last_label)
                
            except Exception as e:
                print(f"⚠️ Erreur: {e}")
                last_label = "error"
                last_conf = 0.0
        
        # Lissage intelligent
        if len(recent_preds) >= 8:
            smoothed = majority_vote(recent_preds)
            if smoothed and smoothed not in ["unknown", "error"]:
                last_label = smoothed
        
        # FPS
        dt = time.time() - t0
        if dt >= 0.5:
            fps = frame_idx / dt
        
        # Interface améliorée
        text1 = f"Signe: {last_label}"
        text2 = f"Confiance: {last_conf*100:.1f}% | FPS: {fps:.1f}"
        text3 = f"Pre-processing: {'ON' if preprocessing_enabled else 'OFF'}"
        text4 = "q=quitter | p=toggle preproc | c=toggle crop"
        
        # Fond texte avec transparence
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (10, 10), (600, 180), (0, 0, 0), -1)
        display_frame = cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0)
        
        # Couleur selon confiance
        if last_conf >= 0.8:
            color = (0, 255, 0)  # Vert
        elif last_conf >= 0.6:
            color = (0, 255, 255)  # Jaune
        else:
            color = (0, 0, 255)  # Rouge
        
        cv2.putText(display_frame, text1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(display_frame, text2, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(display_frame, text3, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(display_frame, text4, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        
        # Indicateur de qualité
        quality_text = f"Qualité: {'HD' if preprocessing_enabled else 'Normal'}"
        cv2.putText(display_frame, quality_text, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # Dessiner la zone de détection
        if crop_box is not None:
            x1, y1, x2, y2 = crop_box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            # Marquer le centre
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(display_frame, (cx, cy), 8, (0, 0, 255), -1)
            cv2.circle(display_frame, (cx, cy), 15, (0, 255, 0), 2)
            # Label
            cv2.putText(display_frame, "ZONE SIGNES", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("Reconnaissance Signes - MODE AMÉLIORÉ", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            preprocessing_enabled = not preprocessing_enabled
            print(f"🔄 Pre-processing: {'ON' if preprocessing_enabled else 'OFF'}")
        if key == ord("c"):
            cropping_enabled = not cropping_enabled
            print(f"🔄 Crop: {'ON' if cropping_enabled else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Session terminée")

if __name__ == "__main__":
    main()
