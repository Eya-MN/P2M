import os
import shutil
import cv2
import numpy as np
import random
from ultralytics import YOLO
import yaml

# Configuration pour améliorer l'accuracy >90%
DATA_DIR = 'yolo_dataset_improved'
OUTPUT_DIR = 'yolo_dataset_ultra'
TARGET_ACCURACY = 90.0

def aggressive_augmentation_for_better_accuracy():
    """Augmentation agressive pour atteindre >90%"""
    
    print("🚀 CRÉATION DATASET ULTRA-AMÉLIORÉ POUR >90%")
    print("=" * 60)
    
    # Créer les dossiers
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)
    
    # Récupérer les classes
    classes = [d for d in os.listdir(os.path.join(DATA_DIR, 'train')) 
               if os.path.isdir(os.path.join(DATA_DIR, 'train', d))]
    classes.sort()
    
    print(f"📋 {len(classes)} classes: {classes}")
    
    def ultra_augment_image(img, class_name):
        """Augmentation ultra-agressive"""
        h, w = img.shape[:2]
        augmented_images = []
        
        # Transformations de base
        transforms = []
        
        # Rotations multiples
        for angle in [-30, -20, -15, -10, -5, 0, 5, 10, 15, 20, 30]:
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            transforms.append(lambda img, M=M: cv2.warpAffine(img, M, (w, h)))
        
        # Zoom multiples
        for scale in [0.7, 0.8, 0.9, 1.1, 1.2, 1.3]:
            new_h, new_w = int(h/scale), int(w/scale)
            transforms.append(lambda img, nh=new_h, nw=new_w: cv2.resize(img, (nw, nh))[int(h-nh)//2:int(h-(h-nh)//2), int(w-nw)//2:int(w-(w-nw)//2)])
        
        # Luminosité extrême
        for brightness in [0.5, 0.6, 0.7, 0.8, 1.2, 1.3, 1.4, 1.5]:
            transforms.append(lambda img, b=brightness: np.clip(img * b, 0, 255).astype(np.uint8))
        
        # Contraste extrême
        for contrast in [0.5, 0.7, 1.3, 1.5, 1.7, 2.0]:
            transforms.append(lambda img, c=contrast: np.clip((img - 128) * c + 128, 0, 255).astype(np.uint8))
        
        # Flips
        transforms.append(lambda img: cv2.flip(img, 1))  # Horizontal
        
        # Bruit gaussien
        for noise_level in [5, 10, 15, 20]:
            transforms.append(lambda img, nl=noise_level: np.clip(img + np.random.normal(0, nl, img.shape), 0, 255).astype(np.uint8))
        
        # Flou léger (pour simuler mouvement)
        for blur_level in [1, 2]:
            transforms.append(lambda img, bl=blur_level: cv2.GaussianBlur(img, (bl*2+1, bl*2+1), 0))
        
        # Applications aléatoires
        num_transforms = random.randint(3, 8)
        selected_transforms = random.sample(transforms, num_transforms)
        
        result = img.copy()
        for transform in selected_transforms:
            try:
                result = transform(result)
            except:
                continue
        
        return result
    
    # Traiter chaque split
    for split in ['train', 'val', 'test']:
        print(f"\n📁 Traitement {split.upper()}:")
        total_images = 0
        
        for class_name in classes:
            class_dir = os.path.join(DATA_DIR, split, class_name)
            output_class_dir = os.path.join(OUTPUT_DIR, split, class_name)
            os.makedirs(output_class_dir, exist_ok=True)
            
            if not os.path.exists(class_dir):
                continue
                
            images = [f for f in os.listdir(class_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            class_total = 0
            
            for img_name in images:
                img_path = os.path.join(class_dir, img_name)
                
                try:
                    # Lire l'image originale
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Copier l'original
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    dest_path = os.path.join(output_class_dir, img_name)
                    cv2.imwrite(dest_path, img_rgb)
                    class_total += 1
                    
                    # Augmentations agressives (uniquement pour train)
                    if split == 'train':
                        # Créer 5 versions augmentées par image
                        for i in range(5):
                            aug_img = ultra_augment_image(img, class_name)
                            aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                            aug_name = f"ultra_{i}_{img_name}"
                            aug_dest = os.path.join(output_class_dir, aug_name)
                            cv2.imwrite(aug_dest, aug_img_bgr)
                            class_total += 1
                
                except Exception as e:
                    print(f"   ⚠️ Erreur {img_name}: {e}")
                    continue
            
            print(f"   ✅ {class_name}: {class_total} images")
            total_images += class_total
        
        print(f"   📊 Total {split}: {total_images} images")
    
    print(f"\n🎯 Dataset ultra-amélioré créé dans: {OUTPUT_DIR}")
    return OUTPUT_DIR

def train_ultra_model():
    """Entraînement ultra-optimisé pour >90%"""
    
    print("\n🎯 ENTRAÎNEMENT ULTRA-OPTIMISÉ POUR >90%")
    print("=" * 60)
    
    # Créer le dataset amélioré
    dataset_path = aggressive_augmentation_for_better_accuracy()
    
    # Utiliser YOLOv8m-cls (medium) pour plus de puissance
    model = YOLO('yolov8m-cls.pt')  # Plus puissant que 's'
    
    print("🚀 Démarrage entraînement ultra-optimisé...")
    
    # Paramètres ultra-optimisés
    results = model.train(
        data=dataset_path,
        epochs=150,              # Plus d'epochs
        imgsz=256,              # Plus grande taille
        batch=16,               # Batch plus petit pour plus de précision
        name='sign_classification_ultra',
        save_period=10,
        plots=True,
        device='cpu',
        
        # Learning rate ultra-optimisé
        lr0=0.0005,             # Plus petit pour plus de précision
        lrf=0.005,              # Décroissance progressive
        momentum=0.937,
        weight_decay=0.0003,      # Plus de régularisation
        warmup_epochs=5,          # Plus d'échauffement
        
        # Data augmentation maximale
        hsv_h=0.02,              # Teinte
        hsv_s=0.8,              # Saturation
        hsv_v=0.5,              # Luminosité
        degrees=20.0,             # Plus de rotation
        translate=0.15,           # Plus de translation
        scale=0.6,               # Plus de zoom
        shear=3.0,               # Plus de cisaillement
        perspective=0.001,        # Perspective légère
        flipud=0.0,
        fliplr=0.5,              # Flip horizontal
        
        # Optimisations avancées
        patience=80,              # Plus de patience
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.1,             # Dropout pour régularisation
        val=True,
        split='val',
        
        # Early stopping intelligent
        exist_ok=False,
        resume=False,
    )
    
    print("✅ Entraînement ultra-optimisé terminé !")
    print(f"📁 Résultats dans: runs/classify/sign_classification_ultra/")
    
    return model, 'runs/classify/sign_classification_ultra/weights/best.pt'

def evaluate_ultra_model(model_path):
    """Évaluation ultra-précise"""
    
    print("\n📊 ÉVALUATION ULTRA-PRÉCISE")
    print("=" * 60)
    
    model = YOLO(model_path)
    
    # Test sur le dataset de test original (non-augmenté)
    test_dir = os.path.join(DATA_DIR, 'test')
    classes = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    classes.sort()
    
    all_images = []
    for class_name in classes:
        class_dir = os.path.join(test_dir, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for img in images:
            all_images.append((os.path.join(class_dir, img), class_name))
    
    print(f"📸 Test sur {len(all_images)} images...")
    
    correct = 0
    total = 0
    class_accuracies = {}
    
    for img_path, true_class in all_images:
        try:
            result = model.predict(img_path, verbose=False)[0]
            predicted_class = result.names[result.probs.top1]
            confidence = float(result.probs.top1conf)
            
            total += 1
            if predicted_class == true_class:
                correct += 1
            
            # Compter par classe
            if true_class not in class_accuracies:
                class_accuracies[true_class] = {'correct': 0, 'total': 0}
            
            class_accuracies[true_class]['total'] += 1
            if predicted_class == true_class:
                class_accuracies[true_class]['correct'] += 1
        
        except Exception as e:
            print(f"⚠️ Erreur {img_path}: {e}")
            continue
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    print(f"\n🎯 RÉSULTATS FINAUX:")
    print(f"   Images testées: {total}")
    print(f"   Correctes: {correct}")
    print(f"   ACCURACY: {accuracy:.2f}%")
    
    if accuracy >= TARGET_ACCURACY:
        print(f"🎉 OBJECTIF ATTEINT ! >{TARGET_ACCURACY}% !!!")
    else:
        print(f"⚠️ Objectif non atteint, il manque {TARGET_ACCURACY - accuracy:.2f}%")
    
    # Top 5 et Bottom 5 classes
    class_results = []
    for cls, stats in class_accuracies.items():
        acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        class_results.append((cls, acc, stats['correct'], stats['total']))
    
    class_results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 TOP 5 CLASSES:")
    for cls, acc, cor, tot in class_results[:5]:
        print(f"   🥇 {cls:15} : {acc:.1f}% ({cor}/{tot})")
    
    print(f"\n❌ 5 CLASSES DIFFICILES:")
    for cls, acc, cor, tot in class_results[-5:]:
        print(f"   ⚠️ {cls:15} : {acc:.1f}% ({cor}/{tot})")
    
    return accuracy

if __name__ == "__main__":
    print("🚀 PROGRAMME ULTRA-OPTIMISATION >90%")
    print("🎯 Reconnaissance des Signes Tunisiens")
    print("=" * 60)
    
    # Étape 1: Entraînement ultra-optimisé
    model, model_path = train_ultra_model()
    
    # Étape 2: Évaluation
    accuracy = evaluate_ultra_model(model_path)
    
    print(f"\n🎉 MISSION TERMINÉE")
    print(f"🎯 Accuracy finale: {accuracy:.2f}%")
    
    if accuracy >= TARGET_ACCURACY:
        print(f"✅ SUCCÈS ! Vous pouvez maintenant utiliser:")
        print(f"   python webcam_recognition.py")
    else:
        print(f"💪 Continuez l'entraînement ou ajustez les paramètres")
