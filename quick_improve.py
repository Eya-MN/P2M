import os
from ultralytics import YOLO

def quick_improve_model():
    """Amélioration rapide du modèle existant"""
    
    print("🚀 AMÉLIORATION RAPIDE DU MODÈLE")
    print("=" * 50)
    print("💡 Stratégie: Fine-tuning du modèle existant")
    print("   - Moins d'epochs (30 au lieu de 150)")
    print("   - Data augmentation légère")
    print("   - Focus sur la précision")
    print("=" * 50)
    
    # Utiliser le modèle existant comme base
    model_path = 'runs/classify/sign_classification_improved2/weights/best.pt'
    
    if not os.path.exists(model_path):
        print("❌ Modèle de base non trouvé")
        return None
    
    print("✅ Chargement modèle existant...")
    model = YOLO(model_path)
    
    # Fine-tuning rapide
    print("🎯 Fine-tuning rapide (30 epochs)...")
    
    results = model.train(
        data='yolo_dataset_improved',
        epochs=30,              # Beaucoup moins
        imgsz=224,              # Taille standard
        batch=32,               # Batch raisonnable
        name='sign_classification_quick',
        save_period=5,
        plots=True,
        device='cpu',
        
        # Learning rate prudent
        lr0=0.0001,             # Très petit pour fine-tuning
        lrf=0.001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=2,
        
        # Data augmentation légère
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=10.0,
        translate=0.05,
        scale=0.3,
        shear=1.0,
        flipud=0.0,
        fliplr=0.5,
        
        # Early stopping
        patience=15,
        val=True,
    )
    
    print("✅ Fine-tuning terminé !")
    print(f"📁 Nouveau modèle: runs/classify/sign_classification_quick/weights/best.pt")
    
    return 'runs/classify/sign_classification_quick/weights/best.pt'

def test_new_model(model_path):
    """Test rapide du nouveau modèle"""
    
    print("\n📊 TEST RAPIDE DU NOUVEAU MODÈLE")
    print("=" * 50)
    
    model = YOLO(model_path)
    
    # Test sur quelques images seulement
    test_dir = 'yolo_dataset_improved/test'
    classes = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))][:5]  # 5 classes seulement
    
    total = 0
    correct = 0
    
    for class_name in classes:
        class_dir = os.path.join(test_dir, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith('.jpg')][:5]  # 5 images par classe
        
        for img_name in images:
            img_path = os.path.join(class_dir, img_name)
            
            try:
                result = model.predict(img_path, verbose=False)[0]
                predicted = result.names[result.probs.top1]
                confidence = float(result.probs.top1conf)
                
                total += 1
                if predicted == class_name:
                    correct += 1
                
                print(f"   {class_name} → {predicted} ({confidence:.2f})")
                
            except Exception as e:
                print(f"   Erreur {img_name}: {e}")
    
    accuracy = (correct / total * 100) if total > 0 else 0
    
    print(f"\n🎯 RÉSULTAT RAPIDE:")
    print(f"   Images testées: {total}")
    print(f"   Correctes: {correct}")
    print(f"   Accuracy: {accuracy:.1f}%")
    
    return accuracy

if __name__ == "__main__":
    print("🎯 AMÉLIORATION RAPIDE - OBJECTIF >90%")
    print("⚡ Temps estimé: 10-15 minutes")
    print("=" * 60)
    
    # Étape 1: Fine-tuning rapide
    new_model_path = quick_improve_model()
    
    if new_model_path:
        # Étape 2: Test rapide
        accuracy = test_new_model(new_model_path)
        
        if accuracy >= 90:
            print(f"\n🎉 SUCCÈS ! Accuracy: {accuracy:.1f}%")
            print(f"✅ Utilisez: python webcam_simple.py")
        else:
            print(f"\n⚠️ Accuracy: {accuracy:.1f}% (objectif 90%)")
            print(f"💪 Le modèle est amélioré mais peut encore progresser")
    else:
        print(f"❌ Échec de l'amélioration")
