import os
import cv2
import numpy as np
from ultralytics import YOLO
import shutil

# --- CONFIGURATION ---
DATA_DIR = r'C:\Users\Administrateur\Documents\P2M\dataset_clean'
CROPPED_DIR = r'C:\Users\Administrateur\Documents\P2M\dataset_cropped'

def detect_and_crop_hands():
    """
    Détecte les mains/haut du corps et crop les images
    """
    print("🔍 DÉTECTION ET CROPPING DES MAINS...")
    
    # Créer le dossier de destination
    if not os.path.exists(CROPPED_DIR):
        os.makedirs(CROPPED_DIR)
    
    # Charger un modèle de détection de personnes
    # On utilise YOLOv8 pour détecter les personnes
    detector = YOLO('yolov8n.pt')  # Modèle de détection d'objets
    
    classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    classes.sort()
    
    total_processed = 0
    total_success = 0
    
    for class_name in classes:
        print(f"\n📁 Traitement de la classe : {class_name}")
        
        # Créer le dossier de classe
        class_dir = os.path.join(DATA_DIR, class_name)
        cropped_class_dir = os.path.join(CROPPED_DIR, class_name)
        os.makedirs(cropped_class_dir, exist_ok=True)
        
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_name in images:
            total_processed += 1
            img_path = os.path.join(class_dir, img_name)
            
            try:
                # Lire l'image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                h, w = img.shape[:2]
                
                # Méthode 1: Détection avec YOLO
                results = detector.predict(img_path, verbose=False, classes=[0])  # classe 0 = personne
                
                if len(results[0].boxes) > 0:
                    # Prendre la détection la plus grande
                    boxes = results[0].boxes
                    largest_box = None
                    max_area = 0
                    
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        area = (x2 - x1) * (y2 - y1)
                        if area > max_area:
                            max_area = area
                            largest_box = (int(x1), int(y1), int(x2), int(y2))
                    
                    if largest_box:
                        x1, y1, x2, y2 = largest_box
                        
                        # Agrandir un peu la zone (10%)
                        margin_x = int((x2 - x1) * 0.1)
                        margin_y = int((y2 - y1) * 0.1)
                        
                        x1 = max(0, x1 - margin_x)
                        y1 = max(0, y1 - margin_y)
                        x2 = min(w, x2 + margin_x)
                        y2 = min(h, y2 + margin_y)
                        
                        # Cropper
                        cropped = img[y1:y2, x1:x2]
                        
                        # Sauvegarder
                        cropped_path = os.path.join(cropped_class_dir, img_name)
                        cv2.imwrite(cropped_path, cropped)
                        total_success += 1
                        continue
                
                # Méthode 2: Cropping intelligent si pas de détection
                # Crop la partie supérieure de l'image (où sont généralement les mains)
                top_crop = img[:int(h * 0.7), :]  # 70% du haut
                
                # Sauvegarder
                cropped_path = os.path.join(cropped_class_dir, img_name)
                cv2.imwrite(cropped_path, top_crop)
                total_success += 1
                
            except Exception as e:
                print(f"   ❌ Erreur sur {img_name}: {e}")
                continue
        
        print(f"   ✅ {class_name}: {len(images)} images traitées")
    
    print(f"\n📊 RÉSULTATS DU CROPPING :")
    print(f"   Images traitées : {total_processed}")
    print(f"   Images réussies : {total_success}")
    print(f"   Taux de succès : {(total_success/total_processed)*100:.1f}%")
    
    return CROPPED_DIR

def train_with_cropped_data():
    """
    Entraîne le modèle avec les images croppées
    """
    print("\n🎯 ENTRAÎNEMENT AVEC DONNÉES CROPPÉES...")
    
    # Préparer les données croppées
    cropped_dir = detect_and_crop_hands()
    
    # Utiliser le script d'entraînement amélioré mais avec les données croppées
    from improved_training import prepare_augmented_dataset, train_improved_model, evaluate_improved_model
    
    # Modifier temporairement le DATA_DIR
    import improved_training
    original_data_dir = improved_training.DATA_DIR
    improved_training.DATA_DIR = cropped_dir
    improved_training.OUTPUT_DIR = r'C:\Users\Administrateur\Documents\P2M\yolo_dataset_cropped'
    
    try:
        # Entraîner avec les données croppées
        model = train_improved_model()
        
        # Évaluer
        accuracy, conf_matrix, class_accs = evaluate_improved_model(model)
        
        print(f"\n🎉 RÉSULTATS AVEC CROPPING :")
        print(f"   Accuracy : {accuracy:.2f}%")
        
        # Comparer avec et sans cropping
        print(f"\n📊 COMPARAISON :")
        print(f"   Sans cropping : 95.52%")
        print(f"   Avec cropping : {accuracy:.2f}%")
        print(f"   Amélioration : {accuracy - 95.52:+.2f}%")
        
        return model, accuracy
        
    finally:
        # Restaurer le DATA_DIR original
        improved_training.DATA_DIR = original_data_dir

if __name__ == "__main__":
    print("🚀 LANCEMENT DU CROPPING + ENTRAÎNEMENT")
    print("=" * 60)
    
    # Étape 1 : Cropping des images
    print("ÉTAPE 1: CROPPING DES IMAGES")
    cropped_dir = detect_and_crop_hands()
    
    print(f"\n✅ Images croppées sauvegardées dans : {cropped_dir}")
    print("📁 Vous pouvez vérifier la qualité des crops manuellement")
    
    # Étape 2 : Entraînement (optionnel)
    response = input("\n🤔 Voulez-vous lancer l'entraînement avec les images croppées ? (o/n): ")
    if response.lower() == 'o':
        model, accuracy = train_with_cropped_data()
    else:
        print("📁 Images croppées prêtes. Lancez 'python add_cropping.py' avec 'o' pour l'entraînement")
