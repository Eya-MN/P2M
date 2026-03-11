# 🖐️ Reconnaissance des Signes Tunisiens

## 🎯 Objectif
Développer un système de reconnaissance des signes tunisiens en temps réel avec une accuracy >90% utilisant YOLOv8 et OpenCV.

## 📊 Résultats Actuels
- **Accuracy finale : 95.80%** ✅ (Objectif >90% atteint!)
- **Modèle :** YOLOv8s-cls ultra-optimisé
- **Dataset :** 26 classes de signes tunisiens
- **Images testées :** 357
- **Prédictions correctes :** 342

## 🏗️ Architecture du Projet

### 📁 Structure des Données
```
yolo_dataset_improved/
├── train/     (4,986 images)
├── val/       (356 images)  
└── test/      (357 images)
```

### 🎯 Classes de Signes (26)
1. armee (armée)
2. baladiya (municipalité)
3. carta (carte)
4. centre (centre)
5. dance (danse)
6. dar (maison)
7. directeur (directeur)
8. entendant (entendant)
9. entikhabet (élections)
10. esm (nom)
11. faux (faux)
12. handicap (handicap)
13. help (aide)
14. jadda (grand-mère)
15. kahwa (café)
16. labes (ça va?)
17. maman (maman)
18. objectif (objectif)
19. officiel (officiel)
20. psychologie (psychologie)
21. septembre (septembre)
22. soeur (sœur)
23. sponsor (sponsor)
24. taxi (taxi)
25. travailler (travailler)
26. universite (université)

## 🚀 Modèles Entraînés

### 🏆 Modèle Principal (95.80% accuracy)
- **Chemin :** `runs/classify/sign_classification_ultra_fast2/weights/best.pt`
- **Architecture :** YOLOv8s-cls
- **Taille :** 10.3 MB
- **Performance :** 19.1ms inference par image

### 📈 Évolution des Performances
1. **Modèle initial :** 85.99% accuracy
2. **Après optimisation :** 95.80% accuracy (+9.81%)

## 🎥 Webcam Reconnaissance

### 📋 Scripts Disponibles
1. **`webcam_simple_clean.py`** - Version principale recommandée
   - Crop centré optimisé pour les mains
   - Interface simple et efficace
   - Lissage des prédictions

2. **`webcam_simple.py`** - Version de base
3. **`test_accuracy_fixed.py`** - Test d'accuracy

### 🎮 Utilisation Webcam
```bash
python webcam_simple_clean.py
```

**Contrôles :**
- `q` : Quitter
- `c` : Toggle crop centré/plein écran

## 🔧 Scripts d'Entraînement

### 📈 Optimisation
1. **`ultra_fast_optimize.py`** - Optimisation ultra-rapide (15 epochs)
2. **`improved_training.py`** - Entraînement amélioré
3. **`add_cropping.py`** - Système de cropping des mains

### 🧪 Tests
1. **`test_accuracy_fixed.py`** - Test accuracy complet
2. **`verify_dataset.py`** - Vérification dataset
3. **`analyze_project.py`** - Analyse complète

## 📊 Métriques de Performance

### 🎯 Accuracy par Classe (Top 5)
| Classe | Accuracy | Images |
|--------|----------|---------|
| armee | 100% | 13 |
| baladiya | 100% | 11 |
| centre | 100% | 15 |
| dance | 100% | 16 |
| *...* | *...* | *...* |

### ⚡ Performance Système
- **CPU :** Intel Core i3-1115G4 @ 3.00GHz
- **RAM :** 8GB
- **Inference time :** 19.1ms/image
- **FPS webcam :** ~30 FPS

## 🛠️ Installation

### 📦 Dépendances
```bash
pip install ultralytics opencv-python numpy pandas torch torchvision
```

### 🎯 Versions Recommandées
- Python 3.11+
- ultralytics 8.4.21+
- opencv-python 4.13.0+
- torch 2.10.0+

## 🔄 Workflow de Développement

### 📅 Timeline
1. **Phase 1** : Dataset preparation & cleaning
2. **Phase 2** : Initial training (85.99%)
3. **Phase 3** : Ultra-optimization (95.80%)
4. **Phase 4** : Webcam integration
5. **Phase 5** : Final testing & validation

### 🎯 Technologies Utilisées
- **Deep Learning** : YOLOv8 Classification
- **Computer Vision** : OpenCV
- **Data Processing** : NumPy, Pandas
- **Model Training** : PyTorch, Ultralytics

## 📝 Rapport Technique

### 🧠 Architecture CNN
- **Backbone** : CSPDarknet
- **Neck** : PANet
- **Head** : Classification head
- **Activation** : SiLU
- **Optimizer** : AdamW

### 📊 Training Configuration
- **Epochs** : 15 (fine-tuning)
- **Batch size** : 64
- **Image size** : 224x224
- **Learning rate** : 0.0002
- **Data augmentation** : Oui (légère)

### 🎯 Hyperparameters Optimisés
```yaml
lr0: 0.0002
lrf: 0.002
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 1
hsv_h: 0.01
hsv_s: 0.3
hsv_v: 0.2
degrees: 5.0
fliplr: 0.5
```

## 🚀 Déploiement

### 📱 Requirements
- Webcam HD
- CPU moderne (i3+)
- 4GB+ RAM
- Python 3.11+

### 🎯 Performance Cible
- **Accuracy** : >90% ✅ (95.80% atteint)
- **Latence** : <100ms ✅ (19.1ms)
- **FPS** : >25 ✅ (~30 FPS)

## 🔍 Tests & Validation

### ✅ Tests Réussis
- [x] Accuracy >90% (95.80%)
- [x] Webcam temps réel
- [x] 26 classes reconnues
- [x] Interface utilisateur fonctionnelle
- [x] Lissage des prédictions

### 🧪 Tests de Régression
- [x] Stabilité du modèle
- [x] Performance CPU acceptable
- [x] Gestion des erreurs
- [x] Interface responsive

## 📈 Améliorations Futures

### 🎯 Prochaines Étapes
1. **Optimisation GPU** pour inference plus rapide
2. **Interface graphique** améliorée
3. **Mode multilingue** (français/arabe)
4. **Export mobile** (Android/iOS)
5. **Cloud deployment**

### 🔬 Recherche
- **Data augmentation** avancée
- **Transfer learning** avec modèles plus grands
- **Ensemble methods** pour accuracy >98%

## 👥 Équipe

- **Développeur** : [Votre Nom]
- **Encadrant** : [Nom Encadrant]
- **Institution** : [Votre Institution]

## 📅 Date
- **Début projet** : [Date]
- **Finalisation** : 11 Mars 2026
- **Version** : 1.0.0

## 📄 Licence
[Type de licence]

---

## 🎉 Succès du Projet

✅ **Objectif >90% atteint** : 95.80% accuracy  
✅ **Webcam fonctionnelle** : Reconnaissance en temps réel  
✅ **26 classes** : Tous les signes tunisiens reconnus  
✅ **Performance optimale** : 19.1ms inference, 30 FPS  
✅ **Interface simple** : Utilisation intuitive  

**Le projet est prêt pour la présentation et le déploiement !** 🚀
