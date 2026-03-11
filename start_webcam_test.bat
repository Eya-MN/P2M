@echo off
echo 🎥 LANCEMENT DU TEST WEBCAM
echo ============================
echo.
echo 🔍 VÉRIFICATION DES PRÉREQUIS:
echo --------------------------------

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python n'est pas installé ou pas dans le PATH
    echo.
    echo 💡 SOLUTIONS:
    echo 1. Téléchargez Python depuis https://python.org
    echo 2. Cochez "Add Python to PATH" pendant l'installation
    echo 3. Redémarrez votre terminal après installation
    echo.
    echo 📦 Une fois Python installé, lancez:
    echo    pip install ultralytics opencv-python
    echo.
    pause
    exit /b 1
)

echo ✅ Python trouvé

python -c "import ultralytics" >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ ultralytics n'est pas installé
    echo 📦 Installation en cours...
    pip install ultralytics
)

python -c "import cv2" >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ opencv-python n'est pas installé
    echo 📦 Installation en cours...
    pip install opencv-python
)

echo.
echo ✅ Dépendances vérifiées
echo.
echo 🎯 LANCEMENT DE LA RECONNAISSANCE WEBCAM:
echo =========================================
echo.
echo 📹 Instructions:
echo - Positionnez vos mains devant la caméra
echo - Faites les signes des 26 mots tunisiens
echo - 'c' = activer/désactiver le cropping
echo - 'q' = quitter
echo.
echo 🚀 Démarrage dans 3 secondes...
timeout /t 3 /nobreak >nul

python webcam_recognition.py

echo.
echo 📊 Test terminé
pause
