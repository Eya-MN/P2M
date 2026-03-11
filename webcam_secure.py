import time
from collections import Counter, deque

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = r"runs\classify\sign_classification_ultra_fast2\weights\best.pt"
CAMERA_INDEX = 0

# Paramètres critiques pour éviter les faux positifs
IMG_SIZE = 224
CONF_THRESHOLD = 0.85  # TRÈS ÉLEVÉ pour éviter les erreurs
SMOOTHING_WINDOW = 10
PREDICT_EVERY_N_FRAMES = 3

# Détection de présence de mains
SKIN_DETECTION_THRESHOLD = 0.02  # 2% minimum de peau
HAND_AREA_MIN = 5000  # Minimum 5000 pixels de mains

def detect_skin_presence(frame):
    """Détecter s'il y a des mains dans l'image"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Masque peau plus précis
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Nettoyage
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    
    # Calculer le pourcentage de peau
    total_pixels = frame.shape[0] * frame.shape[1]
    skin_pixels = cv2.countNonZero(skin_mask)
    skin_ratio = skin_pixels / total_pixels
    
    # Trouver les contours de mains
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    hand_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Minimum 1000 pixels
            hand_area += area
    
    return skin_ratio, hand_area, contours

def is_valid_hand_frame(skin_ratio, hand_area, contours):
    """Vérifier si c'est une frame valide avec des mains"""
    
    # Pas assez de peau
    if skin_ratio < SKIN_DETECTION_THRESHOLD:
        return False, f"Peau insuffisante: {skin_ratio:.3f}"
    
    # Pas assez de surface de mains
    if hand_area < HAND_AREA_MIN:
        return False, f"Mains trop petites: {hand_area} pixels"
    
    # Trop de peau (visage dominant)
    if skin_ratio > 0.15:  # Plus de 15% de peau = probablement visage
        return False, f"Trop de peau: {skin_ratio:.3f} (visage?)"
    
    # Vérifier les ratios des contours
    valid_contours = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            # Ratio typique d'une main
            if 0.3 < aspect_ratio < 2.0:
                valid_contours += 1
    
    if valid_contours == 0:
        return False, "Pas de contours de mains valides"
    
    return True, f"Mains détectées: {valid_contours} zones"

def majority_vote(items: deque[str]) -> str | None:
    if not items:
        return None
    return Counter(items).most_common(1)[0][0]

def main() -> None:
    print("🎥 WEBCAM - DÉTECTION SÉCURISÉE ANTI-FAUX-POSITIFS")
    print("=" * 60)
    print("🛡️ SÉCURITÉ ACTIVÉE:")
    print("   ✅ Détection de présence de mains OBLIGATOIRE")
    print("   ✅ Threshold confiance: 85% (très élevé)")
    print("   ✅ Vérification contours de mains")
    print("   ✅ Rejet automatique des frames sans mains")
    print("=" * 60)
    print("💡 Instructions:")
    print("   - Montrez SEULEMENT vos mains")
    print("   - Évitez de montrer votre visage")
    print("   - Bon éclairage sur les mains")
    print("   - Fond clair et simple")
    print("   - 'q' = quitter")
    print("=" * 60)
    
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
    
    print("✅ Webcam démarrée")
    
    recent_preds: deque[str] = deque(maxlen=SMOOTHING_WINDOW)
    last_label = "AUCUN SIGNE"
    last_conf = 0.0
    hand_status = "EN ATTENTE DES MAINS"
    
    frame_idx = 0
    t0 = time.time()
    fps = 0.0
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        frame_idx += 1
        
        # Détecter la présence de mains
        skin_ratio, hand_area, contours = detect_skin_presence(frame)
        has_hands, status_msg = is_valid_hand_frame(skin_ratio, hand_area, contours)
        hand_status = status_msg
        
        # Classification SEULEMENT si des mains sont détectées
        if has_hands and frame_idx % PREDICT_EVERY_N_FRAMES == 0:
            try:
                result = model.predict(frame, imgsz=IMG_SIZE, verbose=False)[0]
                top1_idx = int(result.probs.top1)
                conf = float(result.probs.top1conf)
                label = result.names[top1_idx]
                
                if conf >= CONF_THRESHOLD:
                    last_label = label
                    last_conf = conf
                    print(f"✅ SIGNE VALIDE: {label} ({conf:.2f})")
                else:
                    last_label = "AUCUN SIGNE"
                    last_conf = conf
                    print(f"❌ Confiance trop basse: {conf:.2f}")
                
                recent_preds.append(last_label)
                
            except Exception as e:
                print(f"⚠️ Erreur: {e}")
                last_label = "ERREUR"
                last_conf = 0.0
        elif not has_hands:
            last_label = "AUCUN SIGNE"
            last_conf = 0.0
            recent_preds.clear()
        
        # Lissage
        if len(recent_preds) >= 5:
            smoothed = majority_vote(recent_preds)
            if smoothed and smoothed not in ["AUCUN SIGNE", "unknown", "ERREUR"]:
                last_label = smoothed
        
        # FPS
        dt = time.time() - t0
        if dt >= 0.5:
            fps = frame_idx / dt
        
        # Interface
        if has_hands:
            text1 = f"Signe: {last_label}"
            color1 = (0, 255, 0) if last_label != "AUCUN SIGNE" else (0, 255, 255)
        else:
            text1 = "❌ MONTREZ VOS MAINS"
            color1 = (0, 0, 255)
        
        text2 = f"Confiance: {last_conf*100:.1f}% | FPS: {fps:.1f}"
        text3 = f"État: {hand_status}"
        text4 = "q=quitter"
        
        # Fond texte
        cv2.rectangle(frame, (10, 10), (600, 180), (0, 0, 0), thickness=-1)
        cv2.putText(frame, text1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color1, 2)
        cv2.putText(frame, text2, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(frame, text3, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(frame, text4, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        
        # Dessiner les contours de mains détectés
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                if 0.3 < aspect_ratio < 2.0:
                    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, f"MAIN", (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Indicateur visuel de détection
        status_color = (0, 255, 0) if has_hands else (0, 0, 255)
        cv2.circle(frame, (frame.shape[1] - 50, 50), 20, status_color, -1)
        cv2.putText(frame, "OK" if has_hands else "NO", 
                   (frame.shape[1] - 70, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        cv2.imshow("Reconnaissance Signes - SÉCURISÉE ANTI-FAUX-POSITIFS", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Session terminée - Sécurité garantie !")

if __name__ == "__main__":
    main()
