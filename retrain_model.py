import os
import shutil
from pathlib import Path
from ultralytics import YOLO

def check_dataset_images():
    """Vérifie si les images du dataset existent réellement"""
    
    print("🔍 VÉRIFICATION DU DATASET")
    print("=" * 50)
    
    dataset_dir = Path("yolo_dataset_improved")
    if not dataset_dir.exists():
        print("❌ Dataset non trouvé")
        return False
    
    # Compter les images réelles
    total_images = 0
    real_images = 0
    
    for split in ['train', 'val', 'test']:
        split_dir = dataset_dir / split
        if split_dir.exists():
            print(f"\n📁 {split.upper()}:")
            
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                    real_count = len([img for img in images if img.stat().st_size > 1000])  # > 1KB
                    total_images += len(images)
                    real_images += real_count
                    
                    if real_count > 0:
                        print(f"   ✅ {class_dir.name}: {real_count} images réelles")
                    else:
                        print(f"   ❌ {class_dir.name}: {len(images)} fichiers (probablement vides)")
    
    print(f"\n📊 BILAN:")
    print(f"   Total fichiers: {total_images}")
    print(f"   Images réelles: {real_images}")
    print(f"   Taux de fichiers réels: {(real_images/total_images*100):.1f}%" if total_images > 0 else "   Aucun fichier")
    
    return real_images > 0

def create_sample_dataset():
    """Crée un dataset d'exemple pour tester l'entraînement"""
    
    print("\n🔧 CRÉATION DATASET DE TEST")
    print("=" * 50)
    
    # Utiliser le modèle YOLOv8n-cls pour créer un dataset de test
    print("📦 Utilisation du modèle pré-entraîné pour test...")
    
    # Créer un petit dataset de test
    test_dir = Path("dataset_test")
    test_dir.mkdir(exist_ok=True)
    
    # Classes de test (réduites)
    test_classes = ['armee', 'baladiya', 'carta', 'centre', 'dance']
    
    for class_name in test_classes:
        class_dir = test_dir / class_name
        class_dir.mkdir(exist_ok=True)
        print(f"   📁 Créé: {class_dir}")
    
    print(f"\n✅ Dataset de test créé: {test_dir}")
    print(f"   📊 5 classes de test")
    print(f"   💡 Ajoutez vos images manuellement ou utilisez un autre dataset")
    
    return test_dir

def train_simple_model():
    """Entraîne un modèle simple pour tester"""
    
    print("\n🎯 ENTRAÎNEMENT MODÈLE DE TEST")
    print("=" * 50)
    
    # Essayer d'abord avec le dataset existant
    if check_dataset_images():
        print("✅ Dataset existant valide, entraînement...")
        dataset_path = "yolo_dataset_improved"
    else:
        print("⚠️ Dataset existant vide, création dataset de test...")
        test_dataset = create_sample_dataset()
        dataset_path = str(test_dataset)
        print("💡 Ajoutez des images dans les dossiers avant l'entraînement")
        return None
    
    try:
        # Charger modèle de classification
        model = YOLO('yolov8n-cls.pt')
        
        # Entraînement rapide pour test
        print("🚀 Lancement entraînement (10 epochs pour test)...")
        
        results = model.train(
            data=dataset_path,
            epochs=10,
            imgsz=224,
            batch=16,
            name='sign_classification_test',
            plots=True,
            device='cpu',
            verbose=True
        )
        
        print("✅ Entraînement terminé!")
        print(f"📁 Résultats dans: runs/classify/sign_classification_test/")
        
        return model
        
    except Exception as e:
        print(f"❌ Erreur entraînement: {e}")
        return None

def main():
    print("🚀 RÉ-ENTRAÎNEMENT DU MODÈLE DE CLASSIFICATION")
    print("🎯 Reconnaissance des Signes Tunisiens")
    print("=" * 60)
    
    # Étape 1: Vérifier le dataset
    has_valid_data = check_dataset_images()
    
    if not has_valid_data:
        print("\n⚠️ DATASET PROBLÉMATIQUE")
        print("💡 SOLUTIONS:")
        print("   1. Ajoutez vos images manuellement dans yolo_dataset_improved/")
        print("   2. Utilisez un dataset externe")
        print("   3. Créez un nouveau dataset avec vos propres images")
        
        response = input("\n🤔 Voulez-vous créer un dataset de test? (o/n): ")
        if response.lower() == 'o':
            create_sample_dataset()
            print("\n📁 Dataset de test créé. Ajoutez-y vos images puis relancez.")
        return
    
    # Étape 2: Entraîner
    print("\n🎯 LANCEMENT DE L'ENTRAÎNEMENT...")
    model = train_simple_model()
    
    if model:
        print("\n🎉 ENTRAÎNEMENT RÉUSSI!")
        print("📋 PROCHAINES ÉTAPES:")
        print("   1. Testez le modèle: python test_accuracy.py")
        print("   2. Lancez la webcam: python webcam_recognition.py")
        print("   3. Évaluez les performances")
    else:
        print("\n❌ ENTRAÎNEMENT ÉCHOUÉ")
        print("💡 Vérifiez le dataset et réessayez")

if __name__ == "__main__":
    main()
