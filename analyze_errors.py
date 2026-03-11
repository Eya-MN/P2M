from ultralytics import YOLO
import os
import cv2
from collections import defaultdict

# Charger le modèle
model = YOLO('runs/classify/sign_classification_improved/weights/best.pt')

# Analyser spécifiquement les erreurs de la classe "faux"
test_dir = 'yolo_dataset_improved/test'
target_class = 'faux'

print(f"🔍 ANALYSE DES ERREURS - CLASSE : {target_class}")
print("=" * 60)

errors = []
class_dir = os.path.join(test_dir, target_class)

for img_name in os.listdir(class_dir):
    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    
    img_path = os.path.join(class_dir, img_name)
    result = model.predict(img_path, verbose=False)[0]
    predicted_class = result.names[result.probs.top1]
    confidence = result.probs.top1conf
    
    if predicted_class != target_class:
        errors.append({
            'image': img_name,
            'predicted': predicted_class,
            'confidence': confidence,
            'path': img_path
        })

print(f"📊 Erreurs trouvées : {len(errors)}/15 images")
print(f"🎯 Accuracy actuelle : {(15-len(errors))/15*100:.1f}%")

print("\n🔍 DÉTAIL DES ERREURS :")
for i, error in enumerate(errors, 1):
    print(f"{i}. {error['image']}")
    print(f"   Prédit : {error['predicted']} (confiance: {error['confidence']:.2f})")
    print("-" * 40)

# Analyser les confusions
confusion_counts = defaultdict(int)
for error in errors:
    confusion_counts[error['predicted']] += 1

print("\n📈 CLASSES CONFUSES AVEC 'faux' :")
for confused_class, count in sorted(confusion_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"   {confused_class}: {count} fois")

# Suggestions d'amélioration
print("\n💡 SUGGESTIONS D'AMÉLIORATION :")
if confusion_counts:
    most_confused = max(confusion_counts, key=confusion_counts.get)
    print(f"1. 'faux' est souvent confondu avec '{most_confused}'")
    print("2. Ajouter plus de variations d'images pour 'faux'")
    print("3. Améliorer l'éclairage et l'angle des photos 'faux'")
    print("4. Data augmentation spécifique pour 'faux'")
else:
    print("✅ Pas d'erreurs trouvées !")
