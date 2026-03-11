import os
import pandas as pd
import yaml
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
import shutil
import cv2
import numpy as np
import random

# --- CONFIGURATION ---
DATA_DIR = r'C:\Users\Administrateur\Documents\P2M\dataset_clean'
OUTPUT_DIR = r'C:\Users\Administrateur\Documents\P2M\yolo_dataset_improved'

def augment_image(img):
    """Applique des augmentations aléatoires à une image"""
    h, w = img.shape[:2]
    
    # Rotation aléatoire (-15° à +15°)
    angle = random.uniform(-15, 15)
    center = (w//2, h//2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img, matrix, (w, h))
    
    # Zoom aléatoire (0.9x à 1.1x)
    zoom = random.uniform(0.9, 1.1)
    new_h, new_w = int(h/zoom), int(w/zoom)
    img = cv2.resize(img, (w, h))
    
    # Luminosité aléatoire
    brightness = random.uniform(0.8, 1.2)
    img = np.clip(img * brightness, 0, 255).astype(np.uint8)
    
    # Flip horizontal (50% de chance)
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
    
    return img

def prepare_augmented_dataset():
    """
    Crée un dataset augmenté avec 3x plus d'images
    """
    print("🚀 Préparation du dataset AUGMENTÉ pour YOLO...")
    
    # Création des dossiers
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)
    
    # Récupération des classes
    classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    classes.sort()
    print(f"📋 Classes trouvées ({len(classes)}): {classes}")
    
    # Division des données originales
    all_images = []
    for class_name in classes:
        class_dir = os.path.join(DATA_DIR, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for img in images:
            all_images.append((os.path.join(class_dir, img), class_name))
    
    # Split : 70% train, 15% val, 15% test
    train_data, temp_data = train_test_split(all_images, test_size=0.3, random_state=42, stratify=[cls for _, cls in all_images])
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42, stratify=[cls for _, cls in temp_data])
    
    # Copie avec augmentation pour le train
    def copy_and_augment_images(image_list, split_name, augment=False):
        copied_count = 0
        for img_path, class_name in image_list:
            # Création du dossier de classe
            class_dir = os.path.join(OUTPUT_DIR, split_name, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Copie de l'original
            img_filename = os.path.basename(img_path)
            dest_img = os.path.join(class_dir, img_filename)
            shutil.copy2(img_path, dest_img)
            copied_count += 1
            
            # Augmentation (uniquement pour train)
            if augment and split_name == 'train':
                # Créer 2 versions augmentées par image
                for i in range(2):
                    # Lecture de l'image
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Application de l'augmentation
                    aug_img = augment_image(img)
                    aug_img = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                    
                    # Sauvegarde
                    aug_filename = f"aug_{i}_{img_filename}"
                    aug_dest = os.path.join(class_dir, aug_filename)
                    cv2.imwrite(aug_dest, aug_img)
                    copied_count += 1
        
        print(f"✅ {split_name}: {copied_count} images (originales + augmentées)")
        return copied_count
    
    train_count = copy_and_augment_images(train_data, 'train', augment=True)
    val_count = copy_and_augment_images(val_data, 'val', augment=False)
    test_count = copy_and_augment_images(test_data, 'test', augment=False)
    
    print(f"\n📊 Répartition AUGMENTÉE des données :")
    print(f"   Train : {train_count} images (~3x plus)")
    print(f"   Val   : {val_count} images")
    print(f"   Test  : {test_count} images")
    
    return OUTPUT_DIR, train_count, val_count, test_count

def train_improved_model():
    """
    Entraîne un modèle amélioré avec optimisations
    """
    print("\n🎯 Démarrage de l'entraînement AMÉLIORÉ...")
    
    # Préparation des données augmentées
    dataset_path, train_count, val_count, test_count = prepare_augmented_dataset()
    
    # Utiliser un modèle plus puissant : yolov8s-cls (small)
    model = YOLO('yolov8s-cls.pt')  # s = small (plus performant que n)
    
    # Entraînement avec paramètres optimisés
    results = model.train(
        data=dataset_path,
        epochs=100,              # Plus d'epochs pour mieux apprendre
        imgsz=224,              # Taille standard
        batch=32,               # Batch plus grand (si RAM permet)
        name='sign_classification_improved',
        save_period=10,         # Sauvegarde toutes les 10 epochs
        plots=True,
        device='cpu',
        
        # Optimisations pour meilleure accuracy
        lr0=0.001,              # Learning rate plus petit
        lrf=0.01,               # Learning rate final
        momentum=0.937,         # Momentum standard
        weight_decay=0.0005,    # Régularisation L2
        warmup_epochs=3,        # Période d'échauffement
        
        # Data augmentation intégrée à YOLO
        hsv_h=0.015,            # Teinte
        hsv_s=0.7,              # Saturation  
        hsv_v=0.4,              # Luminosité
        degrees=15.0,           # Rotation
        translate=0.1,          # Translation
        scale=0.5,              # Zoom
        shear=2.0,              # Cisaillement
        perspective=0.0,        # Perspective
        flipud=0.0,             # Flip vertical
        fliplr=0.5,             # Flip horizontal (50%)
        mosaic=0.0,             # Mosaic (désactivé pour classification)
        mixup=0.0,              # Mixup (désactivé)
        
        # Early stopping
        patience=50,            # Arrêt si pas d'amélioration
    )
    
    print("✅ Entraînement amélioré terminé !")
    print(f"📁 Résultats dans : runs/classify/sign_classification_improved/")
    
    return model

def evaluate_improved_model(model):
    """
    Évaluation détaillée avec matrice de confusion
    """
    print("\n📊 Évaluation du modèle amélioré...")
    
    # Test sur le dataset de test (non-augmenté)
    test_dir = os.path.join(OUTPUT_DIR, 'test')
    results = model.predict(test_dir, save=False)
    
    # Analyse détaillée
    classes = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    classes.sort()
    
    # Matrice de confusion
    confusion_matrix = {cls: {pred_cls: 0 for pred_cls in classes} for cls in classes}
    
    correct = 0
    total = 0
    
    for result in results:
        img_path = result.path
        predicted_class = result.names[result.probs.top1]
        
        # Trouver la vraie classe
        true_class = None
        for class_name in classes:
            if class_name in img_path:
                true_class = class_name
                break
        
        if true_class:
            confusion_matrix[true_class][predicted_class] += 1
            total += 1
            if predicted_class == true_class:
                correct += 1
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    print(f"\n🎯 RÉSULTATS AMÉLIORÉS :")
    print(f"   Images testées : {total}")
    print(f"   Prédictions correctes : {correct}")
    print(f"   ACCURACY : {accuracy:.2f}%")
    
    # Top 5 des meilleures classes
    class_accuracies = {}
    for true_cls in classes:
        class_total = sum(confusion_matrix[true_cls].values())
        class_correct = confusion_matrix[true_cls][true_cls]
        class_acc = (class_correct / class_total * 100) if class_total > 0 else 0
        class_accuracies[true_cls] = class_acc
    
    print(f"\n🏆 Top 5 des classes les mieux reconnues :")
    for cls, acc in sorted(class_accuracies.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"   {cls}: {acc:.1f}%")
    
    print(f"\n📉 5 classes les plus difficiles :")
    for cls, acc in sorted(class_accuracies.items(), key=lambda x: x[1])[:5]:
        print(f"   {cls}: {acc:.1f}%")
    
    return accuracy, confusion_matrix, class_accuracies

if __name__ == "__main__":
    # Étape 1 : Entraînement amélioré
    model = train_improved_model()
    
    # Étape 2 : Évaluation détaillée
    accuracy, conf_matrix, class_accs = evaluate_improved_model(model)
    
    print(f"\n🎉 ACCURACY FINALE : {accuracy:.2f}%")
    
    if accuracy >= 90:
        print("🏆 OBJECTIF ATTEINT ! Votre modèle est excellent !")
    elif accuracy >= 80:
        print("🎯 TRÈS BON ! Presque l'objectif 90% !")
    elif accuracy >= 70:
        print("👍 BON ! En progression significative !")
    else:
        print("💪 Continuez d'optimiser !")
