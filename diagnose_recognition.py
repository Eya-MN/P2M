import os
import cv2
import numpy as np
from collections import Counter
from ultralytics import YOLO

def analyze_recognition_issues():
    """Analyser les problèmes de reconnaissance"""
    
    print("🔍 ANALYSE DES PROBLÈMES DE RECONNAISSANCE")
    print("=" * 60)
    print("📋 Objectif: Comprendre pourquoi la reconnaissance échoue")
    print("=" * 60)
    
    # Charger le modèle
    model_path = 'runs/classify/sign_classification_ultra_fast2/weights/best.pt'
    
    if not os.path.exists(model_path):
        print("❌ Modèle non trouvé, utilisation du modèle de base")
        model_path = 'runs/classify/sign_classification_improved2/weights/best.pt'
    
    try:
        model = YOLO(model_path)
        print(f"✅ Modèle chargé: {model_path}")
    except Exception as e:
        print(f"❌ Erreur modèle: {e}")
        return
    
    # Analyser quelques classes
    test_dir = 'yolo_dataset_improved/test'
    classes = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    classes.sort()
    
    print(f"\n📊 Analyse sur {len(classes)} classes")
    
    # Test sur 3 classes différentes
    test_classes = classes[:3]
    
    for class_name in test_classes:
        print(f"\n🎯 Classe: {class_name}")
        print("-" * 40)
        
        class_dir = os.path.join(test_dir, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith('.jpg')][:5]
        
        predictions = []
        confidences = []
        
        for img_name in images:
            img_path = os.path.join(class_dir, img_name)
            
            try:
                result = model.predict(img_path, verbose=False)[0]
                predicted = result.names[result.probs.top1]
                confidence = float(result.probs.top1conf)
                
                predictions.append(predicted)
                confidences.append(confidence)
                
                print(f"   📸 {img_name[:15]} → {predicted} ({confidence:.2f})")
                
            except Exception as e:
                print(f"   ❌ Erreur {img_name}: {e}")
        
        # Analyse des prédictions
        pred_counter = Counter(predictions)
        avg_conf = np.mean(confidences) if confidences else 0
        
        print(f"\n📈 Résultats pour {class_name}:")
        print(f"   Prédictions: {dict(pred_counter)}")
        print(f"   Confiance moyenne: {avg_conf:.2f}")
        
        correct = sum(1 for p in predictions if p == class_name)
        accuracy = (correct / len(predictions)) * 100 if predictions else 0
        print(f"   Accuracy: {accuracy:.1f}% ({correct}/{len(predictions)})")
        
        if accuracy < 70:
            print(f"   ⚠️ PROBLÈME: Cette classe est mal reconnue !")
    
    print(f"\n🔧 RECOMMANDATIONS:")
    print(f"   1. Si les confiances sont basses (<0.7):")
    print(f"      - Améliorer l'éclairage lors des tests")
    print(f"      - Augmenter le threshold de confiance")
    print(f"   2. Si les prédictions sont fausses:")
    print(f"      - Data augmentation ciblée sur les classes problématiques")
    print(f"      - Plus de données d'entraînement")
    print(f"      - Fine-tuning avec learning rate plus bas")
    print(f"   3. Si les confiances sont élevées mais fausses:")
    print(f"      - Problème de surapprentissage")
    print(f"      - Dataset déséquilibré")
    print(f"      - Labels incorrects dans le dataset")
    
    # Test webcam rapide
    print(f"\n🎥 Test webcam rapide:")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            try:
                result = model.predict(frame, verbose=False)[0]
                predicted = result.names[result.probs.top1]
                confidence = float(result.probs.top1conf)
                print(f"   📹 Webcam → {predicted} ({confidence:.2f})")
            except:
                print(f"   ❌ Erreur webcam")
        cap.release()
    else:
        print(f"   ❌ Webcam non disponible")

if __name__ == "__main__":
    analyze_recognition_issues()
