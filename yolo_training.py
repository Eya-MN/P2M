import os
import pandas as pd
import yaml
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
import shutil

# --- CONFIGURATION ---
DATA_DIR = r'C:\Users\Administrateur\Documents\P2M\dataset_clean'
OUTPUT_DIR = r'C:\Users\Administrateur\Documents\P2M\yolo_dataset'

def prepare_yolo_dataset():
    """
    Prépare le dataset au format YOLO Classification
    YOLO classification a besoin d'une structure simple :
    yolo_dataset/
        train/
            armee/
                image1.jpg
                image2.jpg
                universite/
                ...
        val/
            armee/
            universite/
            ...
        test/
            armee/
            universite/
            ...
    """
    
    print("🚀 Préparation du dataset pour YOLO Classification...")
    
    # Création des dossiers
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)
    
    # Récupération des classes
    classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    classes.sort()
    print(f"📋 Classes trouvées ({len(classes)}): {classes}")
    
    # Division des données
    all_images = []
    for class_name in classes:
        class_dir = os.path.join(DATA_DIR, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for img in images:
            all_images.append((os.path.join(class_dir, img), class_name))
    
    # Split : 70% train, 15% val, 15% test
    train_data, temp_data = train_test_split(all_images, test_size=0.3, random_state=42, stratify=[cls for _, cls in all_images])
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42, stratify=[cls for _, cls in temp_data])
    
    # Copie des fichiers avec structure par classe
    def copy_images(image_list, split_name):
        for img_path, class_name in image_list:
            # Création du dossier de classe
            class_dir = os.path.join(OUTPUT_DIR, split_name, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Copie image
            img_filename = os.path.basename(img_path)
            dest_img = os.path.join(class_dir, img_filename)
            shutil.copy2(img_path, dest_img)
            
        print(f"✅ {split_name}: {len(image_list)} images copiées")
    
    copy_images(train_data, 'train')
    copy_images(val_data, 'val')
    copy_images(test_data, 'test')
    
    print(f"\n📊 Répartition des données :")
    print(f"   Train : {len(train_data)} images ({len(train_data)/len(all_images)*100:.1f}%)")
    print(f"   Val   : {len(val_data)} images ({len(val_data)/len(all_images)*100:.1f}%)")
    print(f"   Test  : {len(test_data)} images ({len(test_data)/len(all_images)*100:.1f}%)")
    
    return OUTPUT_DIR, len(train_data), len(val_data), len(test_data)

def train_yolo_model():
    """
    Entraîne le modèle YOLO pour la classification
    """
    print("\n🎯 Démarrage de l'entraînement YOLO...")
    
    # Préparation des données
    dataset_path, train_count, val_count, test_count = prepare_yolo_dataset()
    
    # Chargement du modèle pré-entraîné
    model = YOLO('yolov8n-cls.pt')  # n = nano (plus rapide), cls = classification
    
    # Entraînement
    results = model.train(
        data=dataset_path,     # Chemin vers le dossier dataset
        epochs=50,           # Nombre de cycles d'entraînement
        imgsz=224,          # Taille des images
        batch=16,           # Images par batch
        name='sign_classification',
        save_period=5,      # Sauvegarde toutes les 5 epochs
        plots=True,         # Génère les graphiques
        device='cpu'        # Utiliser CPU (changer à 'cuda' si GPU disponible)
    )
    
    print("✅ Entraînement terminé !")
    print(f"📁 Résultats sauvegardés dans : runs/classify/sign_classification/")
    
    return model

def evaluate_model(model):
    """
    Évalue le modèle et calcule l'accuracy
    """
    print("\n📊 Évaluation du modèle...")
    
    # Test sur le dataset de test
    test_dir = os.path.join(OUTPUT_DIR, 'test', 'images')
    results = model.predict(test_dir, save=False)
    
    # Comptage des prédictions correctes
    correct = 0
    total = 0
    
    for result in results:
        # Récupération du nom de fichier
        img_path = result.path
        img_name = os.path.basename(img_path)
        
        # Prédiction
        predicted_class = result.names[result.probs.top1]
        
        # Vérité terrain (d'après le nom du dossier original)
        # Comme on a gardé la structure, on peut déduire la classe
        true_class = None
        for class_name in os.listdir(DATA_DIR):
            if img_name.startswith(class_name.split('-')[0]):
                true_class = class_name
                break
        
        if true_class and predicted_class == true_class:
            correct += 1
        total += 1
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    print(f"\n🎯 RÉSULTATS FINAUX :")
    print(f"   Images testées : {total}")
    print(f"   Prédictions correctes : {correct}")
    print(f"   ACCURACY : {accuracy:.2f}%")
    
    return accuracy

if __name__ == "__main__":
    # Étape 1 : Entraînement
    model = train_yolo_model()
    
    # Étape 2 : Évaluation
    accuracy = evaluate_model(model)
    
    print(f"\n🎉 Projet terminé avec une accuracy de {accuracy:.2f}% !")
