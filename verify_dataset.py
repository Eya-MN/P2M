import os
from pathlib import Path

print("🔍 VÉRIFICATION DU DATASET COPIÉ")
print("=" * 50)

test_dir = Path("yolo_dataset_improved/test")
if not test_dir.exists():
    print("❌ Dossier test non trouvé")
    exit(1)

total_images = 0
total_size = 0

print("📁 Images par classe:")
for class_dir in sorted(test_dir.iterdir()):
    if class_dir.is_dir():
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        real_images = [img for img in images if img.stat().st_size > 1000]
        
        if real_images:
            size_sum = sum(img.stat().st_size for img in real_images)
            print(f"   ✅ {class_dir.name}: {len(real_images)} images ({size_sum//1024} KB)")
            total_images += len(real_images)
            total_size += size_sum
        else:
            print(f"   ❌ {class_dir.name}: 0 images réelles")

print(f"\n📊 TOTAL:")
print(f"   Images: {total_images}")
print(f"   Taille: {total_size//1024} KB")

if total_images > 0:
    print(f"\n✅ Dataset valide ! Testons une image...")
    first_image = None
    for class_dir in test_dir.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*.jpg"))
            real_images = [img for img in images if img.stat().st_size > 1000]
            if real_images:
                first_image = real_images[0]
                break
    
    if first_image:
        print(f"   📸 Test: {first_image}")
        print(f"   📏 Taille: {first_image.stat().st_size} bytes")
        
        # Test avec ultralytics
        try:
            from ultralytics import YOLO
            model = YOLO('runs/classify/sign_classification_improved2/weights/best.pt')
            
            print(f"   🎯 Prédiction...")
            results = model.predict(str(first_image), verbose=False)
            result = results[0]
            
            top1_idx = int(result.probs.top1)
            top1_conf = float(result.probs.top1conf)
            class_name = result.names[top1_idx]
            
            print(f"   ✅ Résultat: {class_name} ({top1_conf:.2f})")
            print(f"\n🎉 MODÈLE FONCTIONNEL !")
            
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
else:
    print(f"\n❌ Dataset vide")
