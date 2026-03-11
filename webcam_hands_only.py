import time
from collections import Counter, deque

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = r"runs\classify\sign_classification_improved2\weights\best.pt"
CAMERA_INDEX = 0

# Paramètres optimisés pour détection de mains
IMG_SIZE = 224
CONF_THRESHOLD = 0.70  # Plus élevé pour éviter les faux positifs
SMOOTHING_WINDOW = 10  # Plus de stabilité
PREDICT_EVERY_N_FRAMES = 2

# Détection de peau (pour isoler les mains)
SKIN_LOWER = np.array([0, 20, 70], dtype=np.uint8)
SKIN_UPPER = np.array([20, 255, 255], dtype=np.uint8)

def detect_skin_hands(frame):
    """Détecter les zones de peau (mains)"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Masque de peau
    skin_mask = cv2.inRange(hsv, SKIN_LOWER, SKIN_UPPER)
    
    # Nettoyage du masque
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    
    # Trouver les contours
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrer les contours qui ressemblent à des mains
    hand_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Minimum 1000 pixels
            # Vérifier le ratio (pas trop allongé)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if 0.3 < aspect_ratio < 3.0:  # Ratio raisonnable pour une main
                hand_contours.append((contour, area, (x, y, w, h)))
    
    # Trier par aire et prendre les plus grands
    hand_contours.sort(key=lambda x: x[1], reverse=True)
    
    return hand_contours[:2]  # Maximum 2 mains

def crop_around_hands(frame, hand_contours):
    """Créer un crop autour des mains détectées"""
    if not hand_contours:
        return None, None
    
    h, w = frame.shape[:2]
    
    # Combiner toutes les mains
    all_x = []
    all_y = []
    
    for _, _, (x, y, w, h) in hand_contours:
        all_x.extend([x, x + w])
        all_y.extend([y, y + h])
    
    if not all_x or not all_y:
        return None, None
    
    # Boîte englobante avec marge
    margin = 30
    x1 = max(0, min(all_x) - margin)
    y1 = max(0, min(all_y) - margin)
    x2 = min(w, max(all_x) + margin)
    y2 = min(h, max(all_y) + margin)
    
    # Rendre le crop carré (meilleur pour le CNN)
    crop_w = x2 - x1
    crop_h = y2 - y1
    size = max(crop_w, crop_h)
    
    # Centrer le crop carré
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    
    x1 = max(0, cx - size // 2)
    y1 = max(0, cy - size // 2)
    x2 = min(w, x1 + size)
    y2 = min(h, y1 + size)
    
    roi = frame[y1:y2, x1:x2]
    return roi, (x1, y1, x2, y2)

def majority_vote(items: deque[str]) -> str | None:
    if not items:
        return None
    return Counter(items).most_common(1)[0][0]

def main() -> None:
    print("🎥 DÉTECTION DE SIGNES - MODE MAINS OPTIMISÉ")
    print("=" * 60)
    print("💡 Instructions:")
    print("   - Montrez VOS MAINS clairement")
    print("   - Éloignez votre visage du champ")
    print("   - Bon éclairage sur les mains")
    print("   - Fond clair et simple")
    print("   - 'q' = quitter")
    print("=" * 60)
    
    # Charger le modèle
    try:
        model = YOLO(MODEL_PATH)
        print("✅ Modèle chargé")
    except Exception as e:
        print(f"❌ Erreur modèle: {e}")
        return
    
    # Ouvrir webcam
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
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        frame_idx += 1
        
        # Détecter les mains
        hand_contours = detect_skin_hands(frame)
        
        # Créer le ROI autour des mains
        roi, crop_box = crop_around_hands(frame, hand_contours)
        
        # Classification seulement si on a des mains
        if roi is not None and frame_idx % PREDICT_EVERY_N_FRAMES == 0:
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
        
        # Lissage
        if len(recent_preds) >= 5:
            smoothed = majority_vote(recent_preds)
            if smoothed and smoothed != "unknown":
                last_label = smoothed
        
        # FPS
        dt = time.time() - t0
        if dt >= 0.5:
            fps = frame_idx / dt
        
        # Interface
        text1 = f"Signe: {last_label}"
        text2 = f"Confiance: {last_conf*100:.1f}% | FPS: {fps:.1f}"
        text3 = f"Mains détectées: {len(hand_contours)}"
        text4 = "q=quitter"
        
        # Fond texte
        cv2.rectangle(frame, (10, 10), (500, 160), (0, 0, 0), thickness=-1)
        cv2.putText(frame, text1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, text2, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(frame, text3, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        cv2.putText(frame, text4, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        
        # Dessiner les contours des mains
        for contour, area, (x, y, w, h) in hand_contours:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"Main ({area:.0f})", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Dessiner la zone de crop
        if crop_box is not None:
            x1, y1, x2, y2 = crop_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.circle(frame, ((x1 + x2)//2, (y1 + y2)//2), 10, (0, 0, 255), -1)
        
        cv2.imshow("Détection Signes - MAINS UNIQUEMENT", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Session terminée")

if __name__ == "__main__":
    main()
