import os
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
    
    print("🔍 TEST D'ACCURACY DU MODÈLE ACTUEL")
    print("=" * 60)
    
    # Charger le meilleur modèle
    model_path = r"runs\classify\sign_classification_improved2\weights\best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Modèle non trouvé: {model_path}")
        sys.exit(1)
    
    print(f"✅ Chargement du modèle: {model_path}")
    model = YOLO(model_path)
    
    # Test sur le dataset de test
    test_dir = Path("yolo_dataset_improved/test")
    
    if not test_dir.exists():
        print(f"❌ Dataset de test non trouvé: {test_dir}")
        sys.exit(1)
    
    print(f"✅ Dataset de test trouvé: {test_dir}")
    
    # Lister toutes les images manuellement
    all_images = []
    classes = [d for d in test_dir.iterdir() if d.is_dir()]
    classes.sort()
    
    print(f"📊 {len(classes)} classes trouvées")
    
    for class_dir in classes:
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        real_images = [img for img in images if img.stat().st_size > 1000]
        all_images.extend(real_images)
        print(f"   📁 {class_dir.name}: {len(real_images)} images")
    
    print(f"\n🚀 Lancement des prédictions sur {len(all_images)} images...")
    
    # Analyser les résultats
    confusion_matrix = {cls.name: {pred_cls: 0 for pred_cls in [c.name for c in classes]} for cls in classes}
    correct = 0
    total = 0
    confidence_sum = 0
    
    for i, img_path in enumerate(all_images):
        if i % 50 == 0:
            print(f"   Progression: {i}/{len(all_images)}")
        
        try:
            # Prédiction
            results = model.predict(str(img_path), save=False, verbose=False)
            result = results[0]
            
            predicted_class = result.names[result.probs.top1]
            confidence = float(result.probs.top1conf)
            
            # Trouver la vraie classe
            true_class = img_path.parent.name
            
            confusion_matrix[true_class][predicted_class] += 1
            total += 1
            confidence_sum += confidence
            if predicted_class == true_class:
                correct += 1
                
        except Exception as e:
            print(f"   ⚠️ Erreur sur {img_path.name}: {e}")
            continue
    
    # Calculer l'accuracy
    accuracy = (correct / total) * 100 if total > 0 else 0
    avg_confidence = (confidence_sum / total) * 100 if total > 0 else 0
    
    print("\n" + "=" * 60)
    print("🎯 RÉSULTATS FINAUX")
    print("=" * 60)
    print(f"📊 Images testées     : {total}")
    print(f"✅ Prédictions correctes : {correct}")
    print(f"❌ Prédictions incorrectes : {total - correct}")
    print(f"🎯 ACCURACY        : {accuracy:.2f}%")
    print(f"📈 Confiance moyenne : {avg_confidence:.2f}%")
    
    # Recommandations
    print(f"\n💡 RECOMMANDATIONS:")
    if accuracy >= 90:
        print("   🎉 EXCELLENT ! Objectif >90% atteint !")
        print("   ✅ Le modèle est prêt pour la production")
        print("   🚀 Lancez: python webcam_recognition.py")
    elif accuracy >= 85:
        print("   🎯 TRÈS BON ! Presque l'objectif")
        print("   💡 Entraînement additionnel recommandé")
    elif accuracy >= 80:
        print("   👍 BON ! En progression significative")
        print("   🔧 Optimisations nécessaires")
    else:
        print("   ⚠️ NÉCESSITE AMÉLIORATION")
        print("   🚀 Relancer avec targeted_improvement.py")
    
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("💡 Installez les dépendances: pip install ultralytics opencv-python")
except Exception as e:
    print(f"❌ Erreur: {e}")
    import traceback
    traceback.print_exc()
