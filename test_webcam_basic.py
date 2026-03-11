import os
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
    import cv2
    import time
    
    print("🎥 TEST WEBCAM AVEC MODÈLE PRÉ-ENTRAÎNÉ")
    print("=" * 60)
    
    # Utiliser un modèle pré-entraîné pour tester la webcam
    print("📦 Chargement du modèle YOLOv8n-cls...")
    model = YOLO('yolov8n-cls.pt')  # Modèle de classification pré-entraîné
    
    print("✅ Modèle chargé avec succès")
    print(f"📊 Classes du modèle: {len(model.names)} classes")
    
    # Test avec une image si disponible
    test_images = []
    for root, dirs, files in os.walk("yolo_dataset_improved/test"):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                test_images.append(os.path.join(root, file))
                if len(test_images) >= 3:
                    break
        if len(test_images) >= 3:
            break
    
    if test_images:
        print(f"\n🔍 TEST sur {len(test_images)} images:")
        print("-" * 40)
        
        for i, img_path in enumerate(test_images, 1):
            print(f"📸 Test {i}: {os.path.basename(img_path)}")
            
            # Prédiction
            results = model.predict(img_path, verbose=False)
            result = results[0]
            
            # Afficher les top 5 prédictions
            top5_idx = result.probs.top5
            top5_conf = result.probs.top5conf
            
            print(f"   🎯 Top 5 prédictions:")
            for j, (idx, conf) in enumerate(zip(top5_idx, top5_conf)):
                class_name = model.names[int(idx)]
                print(f"      {j+1}. {class_name}: {conf:.3f}")
            print()
    else:
        print("⚠️  Aucune image de test trouvée")
    
    # Test webcam
    print("🎥 TEST WEBCAM:")
    print("-" * 40)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la webcam")
        print("💡 Vérifiez que votre webcam est connectée")
        sys.exit(1)
    
    print("✅ Webcam ouverte avec succès")
    print("📹 Test de 5 secondes...")
    
    start_time = time.time()
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Afficher le flux
        cv2.imshow('Webcam Test - Press ESC to exit', frame)
        
        # Test de prédiction toutes les secondes
        if frame_count % 30 == 0:  # ~1 seconde à 30 FPS
            try:
                results = model.predict(frame, verbose=False, imgsz=224)
                result = results[0]
                top1_idx = int(result.probs.top1)
                top1_conf = float(result.probs.top1conf)
                class_name = result.names[top1_idx]
                
                print(f"   🎯 Prédiction: {class_name} ({top1_conf:.2f})")
            except Exception as e:
                print(f"   ⚠️  Erreur de prédiction: {e}")
        
        # Quitter après 5 secondes ou avec ESC
        if time.time() - start_time > 5 or cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n📊 RÉSULTATS DU TEST:")
    print(f"   📹 Frames traitées: {frame_count}")
    print(f"   ⏱️  Durée: {time.time() - start_time:.1f}s")
    print(f"   📈 FPS: {frame_count / (time.time() - start_time):.1f}")
    print(f"   ✅ Webcam fonctionnelle")
    
    print(f"\n🎯 PROCHAINES ÉTAPES:")
    print(f"   1. Le système webcam fonctionne")
    print(f"   2. Les modèles pré-entraînés fonctionnent")
    print(f"   3. Il faut ré-entraîner le modèle avec vos données")
    
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("💡 Installez: pip install ultralytics opencv-python")
except Exception as e:
    print(f"❌ Erreur: {e}")
    import traceback
    traceback.print_exc()
