@echo off
echo 🔒 TEST SÉCURITÉ ANTI-FAUX-POSITIFS
echo ==================================
echo.

echo 🎯 OBJECTIF: Éviter les détections incorrectes
echo 💡 Problème: Le modèle détecte même sans mains
echo 🛡️ Solution: Détection de présence de mains OBLIGATOIRE
echo.

echo 📋 TESTS À EFFECTUER:
echo    1. Ne montrez RIEN → doit afficher "MONTREZ VOS MAINS"
echo    2. Montrez votre visage → doit afficher "MONTREZ VOS MAINS"  
echo    3. Montrez vos mains → doit détecter les signes
echo    4. Faites un signe qui n'existe pas → doit afficher "AUCUN SIGNE"
echo.

echo 🚀 LANCEMENT DU TEST...
python webcam_secure.py

echo.
echo ✅ Test terminé - Vérifiez que tous les cas fonctionnent
pause
