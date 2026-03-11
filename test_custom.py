from ultralytics import YOLO
import cv2
import numpy as np
import os

# Charger le modèle
model = YOLO('runs/classify/sign_classification_improved/weights/best.pt')

def test_custom_image(image_path):
    """
    Testez une image personnalisée
    """
    try:
        # Prédiction
        result = model.predict(image_path, verbose=False)[0]
        
        # Résultats
        predicted_class = result.names[result.probs.top1]
        confidence = result.probs.top1conf
        
        # Top 3 prédictions
        top3 = []
        for i in range(3):
            class_id = result.probs.top5[i]
            class_name = result.names[class_id]
            conf = result.probs.top5conf[i].item()
            top3.append(f"{i+1}. {class_name}: {conf:.2f}")
        
        print(f"\n📸 Image : {image_path}")
        print(f"🎯 Prédiction principale : {predicted_class} (confiance: {confidence:.2f})")
        print("🏆 Top 3 prédictions :")
        for pred in top3:
            print(f"   {pred}")
        
        return predicted_class, confidence
        
    except Exception as e:
        print(f"❌ Erreur : {e}")
        return None, None

# Test avec une image du dataset
print("🔍 TEST PERSONNALISÉ")
print("=" * 50)

# Exemple avec une image de test
test_image = "yolo_dataset_improved/test/armee/armee-emna-11.jpg"
if os.path.exists(test_image):
    test_custom_image(test_image)
else:
    print(f"Image non trouvée : {test_image}")
    print("Changez le chemin dans test_image pour tester votre propre image")

print("\n💡 Pour tester votre propre image :")
print("1. Copiez une image dans le dossier P2M")
print("2. Modifiez la variable test_image avec le chemin de votre image")
print("3. Relancez ce script")
