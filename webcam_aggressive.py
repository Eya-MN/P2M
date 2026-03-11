import time
from collections import Counter, deque
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = r"runs\classify\sign_classification_ultra_fast2\weights\best.pt"
CAMERA_INDEX = 0

# Paramètres AGRESSIFS pour reconnaissance
IMG_SIZE = 320  # Beaucoup plus grand
CONF_THRESHOLD = 0.40  # Très bas pour capturer plus
SMOOTHING_WINDOW = 15  # Maximum de stabilité
PREDICT_EVERY_N_FRAMES = 1  # Chaque frame

# Pre-processing AGRESSIF
USE_AGGRESSIVE_PREPROCESSING = True
BRIGHTNESS_FACTOR = 1.5
CONTRAST_FACTOR = 1.3
GAUSSIAN_BLUR = 0.5  # Léger flou pour réduire le bruit
SHARPEN_KERNEL = np.array([[-2,-2,-2], [-2,17,-2], [-2,-2,-2]]) / 9.0

def aggressive_preprocess(frame):
    """Pre-processing agressif pour webcam"""
    
    # Augmentation drastique de la luminosité
    frame = cv2.convertScaleAbs(frame, alpha=BRIGHTNESS_FACTOR, beta=30)
    
    # Contraste maximal
    frame = cv2.convertScaleAbs(frame, alpha=CONTRAST_FACTOR, beta=20)
    
    # Égalisation d'histogramme pour améliorer les détails
    frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    frame_yuv[:,:,0] = cv2.equalizeHist(frame_yuv[:,:,0])
    frame = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2BGR)
    
    # Réduction du bruit
    frame = cv2.fastNlMeansDenoisingColored(frame, None, 5, 5, 7, 21)
    
    # Netteté agressive
    frame = cv2.filter2D(frame, -1, SHARPEN_KERNEL)
    
    # Léger flou gaussien pour lisser
    frame = cv2.GaussianBlur(frame, (3, 3), GAUSSIAN_BLUR)
    
    return frame

def multi_scale_prediction(model, frame):
    """Prédiction multi-échelles pour meilleure reconnaissance"""
    
    scales = [0.8, 1.0, 1.2]
    all_predictions = []
    all_confidences = []
    
    for scale in scales:
        # Redimensionner
        h, w = frame.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        scaled_frame = cv2.resize(frame, (new_w, new_h))
        
        try:
            result = model.predict(scaled_frame, imgsz=IMG_SIZE, verbose=False)[0]
            predicted = result.names[result.probs.top1]
            confidence = float(result.probs.top1conf)
            
            all_predictions.append(predicted)
            all_confidences.append(confidence)
            
        except:
            continue
    
    # Prendre la meilleure prédiction
    if all_confidences:
        best_idx = np.argmax(all_confidences)
        return all_predictions[best_idx], all_confidences[best_idx]
    
    return "unknown", 0.0

def majority_vote(items: deque[str]) -> str | None:
    if not items:
        return None
    return Counter(items).most_common(1)[0][0]

def adaptive_crop(frame):
    """Crop adaptatif qui se concentre sur les mains"""
    h, w = frame.shape[:2]
    
    # Plusieurs zones à tester
    crops = []
    
    # Zone centrale (par défaut)
    crop_h, crop_w = int(h * 0.6), int(w * 0.6)
    start_x, start_y = (w - crop_w) // 2, int(h * 0.35)
    crops.append((max(0, start_x), max(0, start_y), 
                  min(w, start_x + crop_w), min(h, start_y + crop_h)))
    
    # Zone légèrement plus basse (pour les mains)
    start_x, start_y = (w - crop_w) // 2, int(h * 0.45)
    crops.append((max(0, start_x), max(0, start_y), 
                  min(w, start_x + crop_w), min(h, start_y + crop_h)))
    
    # Zone plus large
    crop_h, crop_w = int(h * 0.7), int(w * 0.7)
    start_x, start_y = (w - crop_w) // 2, int(h * 0.3)
    crops.append((max(0, start_x), max(0, start_y), 
                  min(w, start_x + crop_w), min(h, start_y + crop_h)))
    
    return crops

def main() -> None:
    print("🚀 WEBCAM - RECONNAISSANCE AGRESSIVE")
    print("=" * 50)
    print("⚡ MODE AGRESSIF ACTIVÉ:")
    print("   ✅ Pre-processing ultra-optimisé")
    print("   ✅ Multi-échelles")
    print("   ✅ Crops multiples")
    print("   ✅ Stabilisation maximale")
    print("   ✅ Threshold très bas")
    print("=" * 50)
    print("🎯 OBJECTIF: Reconnaissance maximale")
    print("💡 Instructions:")
    print("   - Éclairage MAXIMAL sur les mains")
    print("   - Fond très clair (blanc si possible)")
    print("   - Mains bien visibles, pas d'ombres")
    print("   - 'q' = quitter")
    print("=" * 50)
    
    # Charger le modèle
    try:
        model = YOLO(MODEL_PATH)
        print("✅ Modèle 95.80% chargé")
    except Exception as e:
        print(f"❌ Erreur modèle: {e}")
        return
    
    # Webcam avec qualité maximale
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ Webcam non disponible")
        return
    
    # Configurer webcam pour qualité MAXIMALE
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    
    print("✅ Webcam Full HD démarrée")
    
    recent_preds: deque[str] = deque(maxlen=SMOOTHING_WINDOW)
    last_label = "unknown"
    last_conf = 0.0
    
    frame_idx = 0
    t0 = time.time()
    fps = 0.0
    preprocessing_enabled = USE_AGGRESSIVE_PREPROCESSING
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        frame_idx += 1
        
        # Pre-processing agressif
        display_frame = frame.copy()
        if preprocessing_enabled:
            frame = aggressive_preprocess(frame)
        
        # Test multiple crops
        crops = adaptive_crop(frame)
        best_prediction = "unknown"
        best_confidence = 0.0
        best_crop = None
        
        for i, (x1, y1, x2, y2) in enumerate(crops):
            roi = frame[y1:y2, x1:x2]
            
            # Multi-échelles sur ce crop
            pred, conf = multi_scale_prediction(model, roi)
            
            if conf > best_confidence:
                best_confidence = conf
                best_prediction = pred
                best_crop = (x1, y1, x2, y2)
        
        # Mettre à jour avec la meilleure prédiction
        if best_confidence >= CONF_THRESHOLD:
            last_label = best_prediction
            last_conf = best_confidence
            recent_preds.append(last_label)
        else:
            last_label = "unknown"
            last_conf = best_confidence
        
        # Lissage ultra-stable
        if len(recent_preds) >= 10:
            smoothed = majority_vote(recent_preds)
            if smoothed and smoothed not in ["unknown", "error"]:
                last_label = smoothed
        
        # FPS
        dt = time.time() - t0
        if dt >= 0.5:
            fps = frame_idx / dt
        
        # Interface avec indicateurs de performance
        text1 = f"Signe: {last_label}"
        text2 = f"Confiance: {last_conf*100:.1f}% | FPS: {fps:.1f}"
        text3 = f"Mode: AGRESSIF | Crops: {len(crops)}"
        text4 = f"Pre-processing: {'MAX' if preprocessing_enabled else 'OFF'}"
        
        # Fond avec transparence
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (10, 10), (650, 200), (0, 0, 0), -1)
        display_frame = cv2.addWeighted(overlay, 0.8, display_frame, 0.2, 0)
        
        # Couleur selon performance
        if last_conf >= 0.7:
            color = (0, 255, 0)  # Vert - Excellent
        elif last_conf >= 0.5:
            color = (0, 255, 255)  # Jaune - Acceptable
        else:
            color = (0, 100, 255)  # Orange - Faible mais tenté
        
        cv2.putText(display_frame, text1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(display_frame, text2, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.putText(display_frame, text3, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(display_frame, text4, (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Indicateur de qualité
        if last_conf >= 0.5:
            quality_text = "🎯 RECONNAISSANCE OK"
            qual_color = (0, 255, 0)
        else:
            quality_text = "⚠️ AMÉLIORER ÉCLAIRAGE"
            qual_color = (0, 0, 255)
        
        cv2.putText(display_frame, quality_text, (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.8, qual_color, 2)
        
        # Dessiner toutes les zones testées
        for i, (x1, y1, x2, y2) in enumerate(crops):
            if best_crop == (x1, y1, x2, y2):
                # Meilleur crop en vert
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(display_frame, f"BEST", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # Autres crops en jaune
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
        
        cv2.imshow("🚀 Reconnaissance Signes - MODE AGRESSIF", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            preprocessing_enabled = not preprocessing_enabled
            print(f"🔄 Pre-processing: {'MAX' if preprocessing_enabled else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Session terminée - Mode agressif testé")

if __name__ == "__main__":
    main()
