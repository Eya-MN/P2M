from ultralytics import YOLO
import os

# Charger le meilleur modèle entraîné
model = YOLO('runs/classify/sign_classification_improved/weights/best.pt')

# Évaluer sur le dataset de test
test_dir = 'yolo_dataset_improved/test'
results = model.predict(test_dir, save=False)

# Compter les prédictions correctes
correct = 0
total = 0

print("📊 Évaluation détaillée :")
print("=" * 50)

for result in results:
    img_path = result.path
    predicted_class = result.names[result.probs.top1]
    confidence = result.probs.top1conf
    
    # Extraire la vraie classe du chemin
    true_class = None
    for class_name in os.listdir(test_dir):
        if class_name in img_path:
            true_class = class_name
            break
    
    if true_class:
        status = "✅" if predicted_class == true_class else "❌"
        print(f"{status} {true_class:15} → {predicted_class:15} ({confidence:.2f})")
        
        total += 1
        if predicted_class == true_class:
            correct += 1

accuracy = (correct / total) * 100

print("=" * 50)
print(f"🎯 RÉSULTATS FINAUX :")
print(f"   Images testées : {total}")
print(f"   Prédictions correctes : {correct}")
print(f"   ACCURACY : {accuracy:.2f}%")

if accuracy >= 90:
    print("🏆 OBJECTIF ATTEINT ! Votre modèle est EXCELLENT !")
elif accuracy >= 80:
    print("🎯 TRÈS BON ! Presque l'objectif 90% !")
elif accuracy >= 70:
    print("👍 BON ! En progression significative !")
else:
    print("💪 Continuez d'optimiser !")
