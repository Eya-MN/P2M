import cv2
import time
from ultralytics import YOLO

print("🎥 TEST WEBCAM SIMPLE")
print("=" * 40)

try:
    # Test webcam basique
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la webcam")
        exit(1)
    
    print("✅ Webcam ouverte")
    print("📹 Test de 10 secondes...")
    print("   (appuyez sur ESC pour quitter)")
    
    # Charger modèle YOLO pour détection (pas classification)
    model = YOLO('yolov8n.pt')  # Modèle de détection d'objets
    print("✅ Modèle YOLO chargé")
    
    start_time = time.time()
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Détection toutes les 10 frames
        if frame_count % 10 == 0:
            try:
                results = model.predict(frame, verbose=False)
                result = results[0]
                
                # Dessiner les détections
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    name = model.names[cls]
                    
                    if conf > 0.5:  # Seuil de confiance
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f"{name}: {conf:.2f}", (int(x1), int(y1)-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Afficher FPS
                fps = frame_count / (time.time() - start_time)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
            except Exception as e:
                print(f"⚠️ Erreur détection: {e}")
        
        cv2.imshow('Webcam Test - ESC to exit', frame)
        
        # Quitter après 10 secondes ou avec ESC
        if time.time() - start_time > 10 or cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n📊 RÉSULTATS:")
    print(f"   📹 Frames traitées: {frame_count}")
    print(f"   ⏱️  Durée: {time.time() - start_time:.1f}s")
    print(f"   📈 FPS moyen: {frame_count / (time.time() - start_time):.1f}")
    print(f"   ✅ Webcam et YOLO fonctionnels!")
    
    print(f"\n🎯 PROCHAINES ÉTAPES:")
    print(f"   1. ✅ Python installé")
    print(f"   2. ✅ Dépendances OK")
    print(f"   3. ✅ Webcam fonctionnelle")
    print(f"   4. ✅ YOLO fonctionne")
    print(f"   5. 🔄 Ré-entraîner le modèle de classification")
    
except Exception as e:
    print(f"❌ Erreur: {e}")
    import traceback
    traceback.print_exc()
