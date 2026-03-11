@echo off
echo 📥 TÉLÉCHARGEMENT MODÈLE PUISSANT
echo ==================================
echo.

echo 1️⃣ TÉLÉCHARGEMENT YOLOv8m-cls (plus puissant)...
curl -L -o yolov8m-cls.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt

if %errorlevel% neq 0 (
    echo ❌ Erreur téléchargement yolov8m-cls.pt
    echo.
    echo 2️⃣ TENTATIVE AVEC yolo11s-cls...
    curl -L -o yolo11s-cls.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11s-cls.pt
    
    if %errorlevel% neq 0 (
        echo ❌ Erreur téléchargement yolo11s-cls.pt
        echo.
        echo 3️⃣ UTILISATION yolo11n-cls (le plus rapide)...
        curl -L -o yolo11n-cls.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n-cls.pt
    )
)

echo.
echo ✅ Vérification des fichiers...
for %%f in (*.pt) do (
    echo 📦 %%f : %%~zf bytes
)

echo.
echo 🚀 PRÊT POUR L'ENTRAÎNEMENT RAPIDE !
pause
