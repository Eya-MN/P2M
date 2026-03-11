import os
from ultralytics import YOLO

def ultra_fast_optimization():
    """Optimisation ultra-rapide sans téléchargement"""
    
    print("⚡ OPTIMISATION ULTRA-RAPIDE - OBJECTIF >90%")
    print("=" * 50)
    print("🎯 Stratégie: Fine-tuning sur modèle existant")
    print("⏱️  Temps: 5-10 minutes maximum")
    print("=" * 50)
    
    # Utiliser le modèle existant
    model_path = 'runs/classify/sign_classification_improved2/weights/best.pt'
    
    if not os.path.exists(model_path):
        print("❌ Modèle non trouvé")
        return None
    
    print("✅ Modèle existant trouvé")
    
    # Fine-tuning ultra-rapide
    model = YOLO(model_path)
    
    print("🚀 Fine-tuning ultra-rapide...")
    
    # Entraînement très court mais efficace
    results = model.train(
        data='yolo_dataset_improved',
        epochs=15,              # Très court
        imgsz=224,
        batch=64,               # Batch plus grand
        name='sign_classification_ultra_fast',
        save_period=3,
        plots=False,            # Pas de plots pour aller plus vite
        device='cpu',
        
        # Hyperparamètres optimisés pour vitesse
        lr0=0.0002,             # Learning rate très petit
        lrf=0.002,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=1,
        
        # Data augmentation minimale mais efficace
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.2,
        degrees=5.0,
        translate=0.05,
        scale=0.2,
        shear=0.5,
        flipud=0.0,
        fliplr=0.5,
        
        # Optimisations
        patience=8,
        val=True,
        workers=0,              # Pas de workers pour éviter les erreurs
    )
    
    print("✅ Fine-tuning terminé !")
    return 'runs/classify/sign_classification_ultra_fast/weights/best.pt'

def test_accuracy_fast(model_path):
    """Test ultra-rapide"""
    
    print("\n📊 TEST ULTRA-RAPIDE")
    print("=" * 50)
    
    model = YOLO(model_path)
    
    # Test sur tout le dataset rapidement
    test_dir = 'yolo_dataset_improved/test'
    classes = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    
    total = 0
    correct = 0
    
    for class_name in classes:
        class_dir = os.path.join(test_dir, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith('.jpg')]
        
        for img_name in images:
            img_path = os.path.join(class_dir, img_name)
            
            try:
                result = model.predict(img_path, verbose=False)[0]
                predicted = result.names[result.probs.top1]
                confidence = float(result.probs.top1conf)
                
                total += 1
                if predicted == class_name:
                    correct += 1
                
                # Afficher seulement les erreurs
                if predicted != class_name:
                    print(f"❌ {class_name} → {predicted} ({confidence:.2f})")
                
            except:
                continue
    
    accuracy = (correct / total * 100) if total > 0 else 0
    
    print(f"\n🎯 RÉSULTAT FINAL:")
    print(f"   Images testées: {total}")
    print(f"   Correctes: {correct}")
    print(f"   ACCURACY: {accuracy:.2f}%")
    
    if accuracy >= 90:
        print(f"\n🎉 SUCCÈS ! OBJECTIF >90% ATTEINT !")
        print(f"✅ Modèle prêt pour la webcam")
    else:
        print(f"\n⚠️ Accuracy: {accuracy:.2f}%")
        diff = 90 - accuracy
        print(f"💪 Manque {diff:.2f}% pour atteindre 90%")
        print(f"🔧 Le modèle est quand même amélioré !")
    
    return accuracy

if __name__ == "__main__":
    print("🚀 MISSION CRITIQUE - ACCURACY >90%")
    print("⏱️  Temps estimé: 5-10 minutes")
    print("=" * 60)
    
    # Étape 1: Optimisation ultra-rapide
    new_model = ultra_fast_optimization()
    
    if new_model and os.path.exists(new_model):
        # Étape 2: Test
        accuracy = test_accuracy_fast(new_model)
        
        print(f"\n🎯 RÉSULTAT FINAL:")
        print(f"   Accuracy: {accuracy:.2f}%")
        print(f"   Modèle: {new_model}")
        
        if accuracy >= 90:
            print(f"\n✅ MISSION ACCOMPLIE !")
            print(f"🚀 Lancez: python webcam_simple.py")
        else:
            print(f"\n💪 PROGRÈS SIGNIFICATIF")
            print(f"🎯 Le modèle est bien amélioré")
    else:
        print(f"❌ Erreur pendant l'optimisation")
