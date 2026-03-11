from ultralytics import YOLO
import torch

# Charger le modèle
model = YOLO('runs/classify/sign_classification_improved/weights/best.pt')

# Analyser l'architecture CNN
print("🧠 ARCHITECTURE CNN DU MODÈLE")
print("=" * 60)

# Afficher la structure du modèle
print("📋 Structure complète :")
print(model.model)

print("\n🔍 Détails des couches CNN :")
print("-" * 60)

# Analyser chaque couche
for i, layer in enumerate(model.model.model):
    layer_type = type(layer).__name__
    
    if hasattr(layer, 'conv') or 'Conv' in layer_type:
        print(f"Couche {i}: {layer_type}")
        
        if hasattr(layer, 'conv'):
            conv = layer.conv
            if hasattr(conv, 'out_channels'):
                print(f"   → Filtres: {conv.out_channels}")
            if hasattr(conv, 'kernel_size'):
                print(f"   → Kernel: {conv.kernel_size}")
            if hasattr(conv, 'stride'):
                print(f"   → Stride: {conv.stride}")
        
        print("-" * 40)

# Compter les paramètres
total_params = sum(p.numel() for p in model.model.parameters())
trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)

print(f"\n📊 STATISTIQUES DU CNN :")
print(f"   Paramètres totaux : {total_params:,}")
print(f"   Paramètres entraînables : {trainable_params:,}")
print(f"   Taille du modèle : {total_params * 4 / 1024 / 1024:.1f} MB")

# Vérifier les couches de convolution
conv_layers = 0
for module in model.model.modules():
    if hasattr(module, 'kernel_size'):  # C'est une couche de convolution
        conv_layers += 1

print(f"   Couches de convolution : {conv_layers}")

print(f"\n🎯 TYPE DE RÉSEAU :")
print(f"   ✅ Convolutional Neural Network (CNN)")
print(f"   ✅ Deep Learning")
print(f"   ✅ Computer Vision")

print(f"\n🔬 FONCTIONNEMENT DU CNN :")
print(f"   1. Input : Image 224x224x3 (RGB)")
print(f"   2. Couches Conv : Extraction de features")
print(f"   3. Pooling : Réduction dimensionnelle")
print(f"   4. Fully Connected : Classification finale")
print(f"   5. Output : 26 classes de signes")
