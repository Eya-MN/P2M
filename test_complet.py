from ultralytics import YOLO
import os
from collections import defaultdict

# Charger le modèle
model = YOLO('runs/classify/sign_classification_improved/weights/best.pt')

# Dossier de test
test_dir = 'yolo_dataset_improved/test'

print("📊 TEST COMPLET DU MODÈLE")
print("=" * 60)

# Statistiques par classe
class_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
total_correct = 0
total_images = 0

print("Analyse de toutes les images de test...")

# Parcourir toutes les classes et images
for class_name in sorted(os.listdir(test_dir)):
    class_dir = os.path.join(test_dir, class_name)
    if not os.path.isdir(class_dir):
        continue
    
    # Tester toutes les images de cette classe
    for img_name in os.listdir(class_dir):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        img_path = os.path.join(class_dir, img_name)
        
        # Prédiction
        result = model.predict(img_path, verbose=False)[0]
        predicted_class = result.names[result.probs.top1]
        
        # Compter
        class_stats[class_name]['total'] += 1
        total_images += 1
        
        if predicted_class == class_name:
            class_stats[class_name]['correct'] += 1
            total_correct += 1

# Calculer l'accuracy globale
global_accuracy = (total_correct / total_images) * 100

print(f"\n🎯 ACCURACY GLOBALE : {global_accuracy:.2f}% ({total_correct}/{total_images})")
print("=" * 60)

# Accuracy par classe
print("📊 PERFORMANCE PAR CLASSE :")
print("-" * 60)

class_accuracies = []
for class_name, stats in class_stats.items():
    accuracy = (stats['correct'] / stats['total']) * 100
    class_accuracies.append((class_name, accuracy, stats['correct'], stats['total']))

# Trier par accuracy
class_accuracies.sort(key=lambda x: x[1], reverse=True)

for class_name, accuracy, correct, total in class_accuracies:
    bar = "█" * int(accuracy / 5) + "░" * (20 - int(accuracy / 5))
    print(f"{class_name:15} : {accuracy:5.1f}% ({correct:2d}/{total:2d}) [{bar}]")

# Statistiques
accuracies = [acc for _, acc, _, _ in class_accuracies]
print(f"\n📈 STATISTIQUES :")
print(f"   Moyenne par classe : {sum(accuracies)/len(accuracies):.1f}%")
print(f"   Meilleure classe : {max(accuracies):.1f}%")
print(f"   Moins bonne classe : {min(accuracies):.1f}%")

# Validation
if global_accuracy >= 95:
    print("\n🏆 EXCEPTIONNEL ! Votre modèle est excellent !")
elif global_accuracy >= 90:
    print("\n🎯 TRÈS BON ! Objectif dépassé !")
elif global_accuracy >= 80:
    print("\n👍 BON ! Performance solide !")
else:
    print("\n💪 Continuez d'optimiser !")
