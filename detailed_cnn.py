from ultralytics import YOLO
import torch.nn as nn

# Charger le modèle
model = YOLO('runs/classify/sign_classification_improved/weights/best.pt')

print("🧠 ARCHITECTURE DÉTAILLÉE DU CNN")
print("=" * 70)

layer_count = 0
conv_layers = []
activation_layers = []
batchnorm_layers = []
pooling_layers = []
linear_layers = []

def analyze_layer(module, depth=0):
    global layer_count
    
    layer_type = type(module).__name__
    layer_count += 1
    
    indent = "  " * depth
    
    # Classification des couches
    if 'Conv' in layer_type:
        conv_layers.append((layer_count, layer_type, module))
        print(f"Couche {layer_count:2d}: {indent}🔲 {layer_type}")
        
        if hasattr(module, 'conv'):
            conv = module.conv
            if hasattr(conv, 'out_channels') and hasattr(conv, 'kernel_size'):
                print(f"         └─ Filtres: {conv.out_channels}, Kernel: {conv.kernel_size}")
                
    elif 'BatchNorm' in layer_type:
        batchnorm_layers.append((layer_count, layer_type))
        print(f"Couche {layer_count:2d}: {indent}📊 {layer_type}")
        
    elif 'SiLU' in layer_type or 'ReLU' in layer_type or 'Act' in layer_type:
        activation_layers.append((layer_count, layer_type))
        print(f"Couche {layer_count:2d}: {indent}⚡ {layer_type}")
        
    elif 'Pool' in layer_type or 'AdaptiveAvgPool' in layer_type:
        pooling_layers.append((layer_count, layer_type))
        print(f"Couche {layer_count:2d}: {indent}📉 {layer_type}")
        
    elif 'Linear' in layer_type or 'Dropout' in layer_type:
        linear_layers.append((layer_count, layer_type))
        print(f"Couche {layer_count:2d}: {indent}🔗 {layer_type}")
        if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
            print(f"         └─ {module.in_features} → {module.out_features}")
    
    elif 'C2f' in layer_type or 'Bottleneck' in layer_type or 'SPPF' in layer_type:
        print(f"Couche {layer_count:2d}: {indent}🏗️ {layer_type} (Bloc composite)")
        
    else:
        print(f"Couche {layer_count:2d}: {indent}📦 {layer_type}")
    
    # Analyser les sous-couches
    if hasattr(module, 'children'):
        for child in module.children():
            analyze_layer(child, depth + 1)

# Analyser l'architecture complète
print("🔍 STRUCTURE COMPLÈTE :")
print("-" * 70)

analyze_layer(model.model)

print("\n" + "=" * 70)
print("📊 RÉCAPITULATIF DES COUCHES :")
print("-" * 70)

print(f"🔲 Couches de Convolution : {len(conv_layers)}")
for i, (num, name, layer) in enumerate(conv_layers, 1):
    print(f"   {i}. Couche {num}: {name}")

print(f"\n📊 Couches de BatchNorm : {len(batchnorm_layers)}")
print(f"⚡ Couches d'Activation : {len(activation_layers)}")
print(f"📉 Couches de Pooling : {len(pooling_layers)}")
print(f"🔗 Couches Fully Connected : {len(linear_layers)}")

print(f"\n🏗️ Blocs Spéciaux :")
print(f"   • C2f : Cross Stage Partial (moderne)")
print(f"   • Bottleneck : Residual connections")
print(f"   • SPPF : Spatial Pyramid Pooling Fast")

print(f"\n🎯 TYPE D'ARCHITECTURE :")
print(f"   ✅ CSPNet (Cross Stage Partial Network)")
print(f"   ✅ Deep Residual Learning")
print(f"   ✅ Modern CNN Architecture (2023)")

print(f"\n📈 CARACTÉRISTIQUES PRINCIPALES :")
print(f"   • {len(conv_layers)} couches de convolution principales")
print(f"   • Connections résiduelles (ResNet-like)")
print(f"   • Attention mechanisms implicites")
print(f"   • Efficient feature extraction")

# Calculer la profondeur réelle
def count_conv_layers(module):
    count = 0
    if isinstance(module, nn.Conv2d):
        count += 1
    for child in module.children():
        count += count_conv_layers(child)
    return count

real_conv_depth = count_conv_layers(model.model)
print(f"\n🔬 PROFONDEUR RÉELLE :")
print(f"   • Couches Conv2d : {real_conv_depth}")
print(f"   • Couches totales : {layer_count}")
print(f"   • Profondeur effective : ~{real_conv_depth + len(linear_layers)}")
