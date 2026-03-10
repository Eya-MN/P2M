import os
import shutil
import cv2
import numpy as np
import random
from ultralytics import YOLO
import yaml

# --- CONFIGURATION ---
DATA_DIR = r'C:\Users\Administrateur\Documents\P2M\dataset_clean'
OUTPUT_DIR = r'C:\Users\Administrateur\Documents\P2M\yolo_dataset_targeted'
TARGET_CLASSES = ['faux', 'kahwa', 'entendant']  # Classes à améliorer

def aggressive_augment_image(img, target_class):
    """Augmentation agressive pour les classes problématiques"""
    h, w = img.shape[:2]
    
    # Transformations plus intenses pour les classes ciblées
    if target_class == 'faux':
        # 'faux' est confondu avec 'kahwa' -> augmenter les différences
        transformations = [
            # Rotation extrême
            lambda img: cv2.warpAffine(img, cv2.getRotationMatrix2D((w//2, h//2), random.uniform(-30, 30), 1.0), (w, h)),
            # Zoom extrême
            lambda img: cv2.resize(img[random.randint(10, 50):h-random.randint(10, 50), 
                                         random.randint(10, 50):w-random.randint(10, 50)], (w, h)),
            # Changement de luminosité drastique
            lambda img: np.clip(img * random.uniform(0.5, 1.5), 0, 255).astype(np.uint8),
            # Flip horizontal et vertical
            lambda img: cv2.flip(cv2.flip(img, 1), 0),
            # Contraste extrême
            lambda img: np.clip((img - 128) * random.uniform(1.5, 2.5) + 128, 0, 255).astype(np.uint8),
            # Bruit
            lambda img: np.clip(img + np.random.normal(0, 20, img.shape), 0, 255).astype(np.uint8),
        ]
    else:
        # Augmentation standard pour autres classes
        transformations = [
            lambda img: cv2.warpAffine(img, cv2.getRotationMatrix2D((w//2, h//2), random.uniform(-15, 15), 1.0), (w, h)),
            lambda img: cv2.resize(img[random.randint(20, 40):h-random.randint(20, 40), 
                                         random.randint(20, 40):w-random.randint(20, 40)], (w, h)),
            lambda img: np.clip(img * random.uniform(0.8, 1.2), 0, 255).astype(np.uint8),
            lambda img: cv2.flip(img, 1),
        ]
    
    # Appliquer 2-3 transformations aléatoires
    num_transforms = random.randint(2, min(4, len(transformations)))
    selected_transforms = random.sample(transformations, num_transforms)
    
    result = img.copy()
    for transform in selected_transforms:
        result = transform(result)
    
    return result

def prepare_targeted_dataset():
    """
    Crée un dataset avec augmentation ciblée pour les classes problématiques
    """
    print("🚀 Préparation du dataset avec AMÉLIORATION CIBLÉE...")
    
    # Création des dossiers
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)
    
    # Récupération des classes
    classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    classes.sort()
    print(f"📋 Classes trouvées ({len(classes)}): {classes}")
    print(f"🎯 Classes ciblées pour amélioration: {TARGET_CLASSES}")
    
    # Division des données
    all_images = []
    for class_name in classes:
        class_dir = os.path.join(DATA_DIR, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for img in images:
            all_images.append((os.path.join(class_dir, img), class_name))
    
    # Split : 70% train, 15% val, 15% test
    from sklearn.model_selection import train_test_split
    train_data, temp_data = train_test_split(all_images, test_size=0.3, random_state=42, stratify=[cls for _, cls in all_images])
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42, stratify=[cls for _, cls in temp_data])
    
    # Copie avec augmentation ciblée
    def copy_with_targeted_augmentation(image_list, split_name):
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
            
            # Augmentation ciblée pour le train
            if split_name == 'train':
                if class_name in TARGET_CLASSES:
                    # Augmentation agressive : 5x plus d'images pour classes ciblées
                    num_augmented = 5
                else:
                    # Augmentation standard : 2x plus d'images
                    num_augmented = 2
                
                # Lecture de l'image
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Créer les versions augmentées
                for i in range(num_augmented):
                    aug_img = aggressive_augment_image(img, class_name)
                    aug_img = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                    
                    # Sauvegarde
                    aug_filename = f"aug_{i}_{img_filename}"
                    aug_dest = os.path.join(class_dir, aug_filename)
                    cv2.imwrite(aug_dest, aug_img)
                    copied_count += 1
        
        print(f"✅ {split_name}: {copied_count} images totales")
        return copied_count
    
    train_count = copy_with_targeted_augmentation(train_data, 'train')
    val_count = copy_with_targeted_augmentation(val_data, 'val')
    test_count = copy_with_targeted_augmentation(test_data, 'test')
    
    print(f"\n📊 Répartition AMÉLIORÉE des données :")
    print(f"   Train : {train_count} images")
    print(f"   Val   : {val_count} images")
    print(f"   Test  : {test_count} images")
    
    return OUTPUT_DIR, train_count, val_count, test_count

def train_targeted_model():
    """
    Entraîne un modèle avec focus sur les classes problématiques
    """
    print("\n🎯 Démarrage de l'entraînement CIBLÉ...")
    
    # Préparation des données
    dataset_path, train_count, val_count, test_count = prepare_targeted_dataset()
    
    # Utiliser un modèle plus puissant
    model = YOLO('yolov8s-cls.pt')
    
    # Entraînement avec paramètres optimisés pour les classes difficiles
    results = model.train(
        data=dataset_path,
        epochs=150,              # Plus d'epochs pour les classes difficiles
        imgsz=224,
        batch=32,
        name='sign_classification_targeted',
        save_period=15,
        plots=True,
        device='cpu',
        
        # Learning rate plus petit pour plus de précision
        lr0=0.0005,              # Encore plus petit
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        
        # Data augmentation plus agressive
        hsv_h=0.02,              # Plus de variation de teinte
        hsv_s=0.8,               # Plus de saturation
        hsv_v=0.5,               # Plus de variation de luminosité
        degrees=20.0,            # Plus de rotation
        translate=0.15,          # Plus de translation
        scale=0.6,               # Plus de zoom
        shear=5.0,               # Plus de cisaillement
        fliplr=0.6,              # Plus de flips horizontaux
        
        # Early stopping plus patient
        patience=75,
    )
    
    print("✅ Entraînement ciblé terminé !")
    print(f"📁 Résultats dans : runs/classify/sign_classification_targeted/")
    
    return model

def evaluate_targeted_model(model):
    """
    Évaluation spécifique des classes ciblées
    """
    print("\n📊 Évaluation du modèle ciblé...")
    
    # Test sur le dataset de test
    test_dir = os.path.join(OUTPUT_DIR, 'test')
    
    # Évaluation par classe
    class_results = {}
    classes = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    
    for class_name in classes:
        class_dir = os.path.join(test_dir, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        correct = 0
        total = 0
        
        for img_name in images:
            img_path = os.path.join(class_dir, img_name)
            result = model.predict(img_path, verbose=False)[0]
            predicted_class = result.names[result.probs.top1]
            
            total += 1
            if predicted_class == class_name:
                correct += 1
        
        accuracy = (correct / total) * 100 if total > 0 else 0
        class_results[class_name] = {'accuracy': accuracy, 'correct': correct, 'total': total}
    
    # Affichage des résultats
    print("\n🎯 RÉSULTATS PAR CLASSE :")
    print("=" * 60)
    
    # Trier par accuracy
    sorted_classes = sorted(class_results.items(), key=lambda x: x[1]['accuracy'])
    
    all_above_90 = True
    for class_name, stats in sorted_classes:
        accuracy = stats['accuracy']
        status = "🏆" if accuracy >= 90 else "⚠️" if accuracy >= 80 else "❌"
        print(f"{status} {class_name:15} : {accuracy:5.1f}% ({stats['correct']}/{stats['total']})")
        
        if accuracy < 90:
            all_above_90 = False
    
    print("=" * 60)
    
    # Focus sur les classes ciblées
    print(f"\n🎯 CLASSES CIBLÉES :")
    for target_class in TARGET_CLASSES:
        if target_class in class_results:
            stats = class_results[target_class]
            print(f"   {target_class}: {stats['accuracy']:.1f}% ({stats['correct']}/{stats['total']})")
    
    # Vérification de l'objectif
    if all_above_90:
        print(f"\n🎉 OBJECTIF ATTEINT ! Toutes les classes > 90% !")
    else:
        below_90 = [cls for cls, stats in class_results.items() if stats['accuracy'] < 90]
        print(f"\n⚠️ Classes encore en dessous de 90% : {below_90}")
    
    return class_results, all_above_90

if __name__ == "__main__":
    # Étape 1 : Entraînement ciblé
    model = train_targeted_model()
    
    # Étape 2 : Évaluation ciblée
    class_results, all_above_90 = evaluate_targeted_model(model)
    
    print(f"\n🎉 MISSION TERMINÉE !")
    print(f"   Objectif 'toutes classes > 90%' : {'ATTEINT ✅' if all_above_90 else 'EN PROGRESSION 🔄'}")
