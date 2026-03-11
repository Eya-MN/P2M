import os
import sys
from pathlib import Path

# Ajouter le répertoire du projet au path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    from ultralytics import YOLO
    
    print("🔍 TEST D'ACCURACY DU MODÈLE ACTUEL")
    print("=" * 60)
    
    # Charger le meilleur modèle
    model_path = r"runs\classify\sign_classification_improved2\weights\best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Modèle non trouvé: {model_path}")
        print("📁 Modèles disponibles:")
        for root, dirs, files in os.walk("runs/classify"):
            for file in files:
                if file == "best.pt":
                    print(f"   - {os.path.join(root, file)}")
        sys.exit(1)
    
    print(f"✅ Chargement du modèle: {model_path}")
    model = YOLO(model_path)
    
    # Test sur le dataset de test
    test_dir = "yolo_dataset_improved/test"
    
    if not os.path.exists(test_dir):
        print(f"❌ Dataset de test non trouvé: {test_dir}")
        sys.exit(1)
    
    print(f"✅ Dataset de test trouvé: {test_dir}")
    
    # Lancer la prédiction sur tout le dataset de test
    print("🚀 Lancement des prédictions...")
    results = model.predict(test_dir, save=False, verbose=False)
    
    # Analyser les résultats
    classes = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    classes.sort()
    
    print(f"📊 {len(classes)} classes trouvées: {classes}")
    
    # Matrice de confusion et comptage
    confusion_matrix = {cls: {pred_cls: 0 for pred_cls in classes} for cls in classes}
    correct = 0
    total = 0
    confidence_sum = 0
    
    print("📈 Analyse des résultats...")
    
    for i, result in enumerate(results):
        if i % 50 == 0:
            print(f"   Progression: {i}/{len(results)}")
        
        img_path = result.path
        predicted_class = result.names[result.probs.top1]
        confidence = float(result.probs.top1conf)
        
        # Trouver la vraie classe
        true_class = None
        for class_name in classes:
            if class_name in img_path:
                true_class = class_name
                break
        
        if true_class:
            confusion_matrix[true_class][predicted_class] += 1
            total += 1
            confidence_sum += confidence
            if predicted_class == true_class:
                correct += 1
    
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
    
    # Accuracy par classe
    print(f"\n📋 ACCURACY PAR CLASSE:")
    print("-" * 40)
    class_accuracies = {}
    for true_cls in classes:
        class_total = sum(confusion_matrix[true_cls].values())
        class_correct = confusion_matrix[true_cls][true_cls]
        class_acc = (class_correct / class_total * 100) if class_total > 0 else 0
        class_accuracies[true_cls] = class_acc
        status = "🏆" if class_acc >= 90 else "👍" if class_acc >= 80 else "⚠️" if class_acc >= 70 else "❌"
        print(f"{status} {true_cls:15} : {class_acc:5.1f}% ({class_correct}/{class_total})")
    
    # Top 5 et Bottom 5
    print(f"\n🏆 TOP 5 CLASSES LES MIEUX RECONNUES:")
    for cls, acc in sorted(class_accuracies.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"   🥇 {cls:15} : {acc:.1f}%")
    
    print(f"\n❌ 5 CLASSES LES PLUS DIFFICILES:")
    for cls, acc in sorted(class_accuracies.items(), key=lambda x: x[1])[:5]:
        print(f"   ⚠️ {cls:15} : {acc:.1f}%")
    
    # Recommandations
    print(f"\n💡 RECOMMANDATIONS:")
    if accuracy >= 90:
        print("   🎉 EXCELLENT ! Objectif >90% atteint !")
        print("   ✅ Le modèle est prêt pour la production")
    elif accuracy >= 85:
        print("   🎯 TRÈS BON ! Presque l'objectif")
        print("   💡 Entraînement additionnel recommandé")
    elif accuracy >= 80:
        print("   👍 BON ! En progression significative")
        print("   🔧 Optimisations nécessaires")
    else:
        print("   ⚠️ NÉCESSITE AMÉLIORATION")
        print("   🚀 Relancer avec targeted_improvement.py")
    
    print(f"\n🚀 PROCHAINE ÉTAPE: Test webcam avec python webcam_recognition.py")
    
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("💡 Installez les dépendances: pip install ultralytics opencv-python")
except Exception as e:
    print(f"❌ Erreur: {e}")
    import traceback
    traceback.print_exc()
