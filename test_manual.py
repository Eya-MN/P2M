from ultralytics import YOLO
import cv2
import os

# Charger le modèle entraîné
model = YOLO('runs/classify/sign_classification_improved/weights/best.pt')

# Dossier de test
test_dir = 'yolo_dataset_improved/test'

print("🔍 TEST MANUEL DU MODÈLE")
print("=" * 60)

# Tester 10 images aléatoires
import random
all_images = []

for class_name in os.listdir(test_dir):
    class_dir = os.path.join(test_dir, class_name)
    if os.path.isdir(class_dir):
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append((os.path.join(class_dir, img_name), class_name))

# Sélectionner 10 images aléatoires
test_images = random.sample(all_images, min(10, len(all_images)))

correct = 0
total = 0

for img_path, true_class in test_images:
    # Prédiction
    result = model.predict(img_path, verbose=False)[0]
    predicted_class = result.names[result.probs.top1]
    confidence = result.probs.top1conf
    
    # Affichage
    status = "✅ CORRECT" if predicted_class == true_class else "❌ INCORRECT"
    print(f"{status}")
    print(f"   Image : {os.path.basename(img_path)}")
    print(f"   Vraie classe : {true_class}")
    print(f"   Prédiction : {predicted_class} (confiance: {confidence:.2f})")
    print("-" * 40)
    
    total += 1
    if predicted_class == true_class:
        correct += 1

accuracy = (correct / total) * 100
print(f"\n🎯 RÉSULTAT DU TEST MANUEL : {accuracy:.1f}% ({correct}/{total})")

if accuracy >= 90:
    print("🏆 EXCELLENT ! Le modèle fonctionne très bien !")
elif accuracy >= 70:
    print("👍 BON ! Performance correcte")
else:
    print("💪 Nécessite peut-être des améliorations")
