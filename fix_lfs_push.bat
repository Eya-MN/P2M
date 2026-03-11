@echo off
echo 🔧 SOLUTION PUSH GITHUB AVEC LFS
echo ================================
echo.
echo PROBLÈME IDENTIFIÉ:
echo ❌ Git LFS bloque le push (fichiers volumineux)
echo ❌ Connexion GitHub instable
echo.
echo SOLUTIONS:
echo.
echo 1️⃣ DÉSACTIVER LFS temporairement:
echo    git lfs uninstall
echo    git push origin master
echo.
echo 2️⃣ PUSH SANS LES FICHIERS VOLUMINEUX:
echo    git lfs ls-files
echo    git rm --cached runs/classify/sign_classification_ultra_fast2/weights/best.pt
echo    git add .gitignore
echo    git commit -m "Remove LFS files for push"
echo    git push origin master
echo.
echo 3️⃣ UTILISER LE SITE WEB:
echo    Allez sur https://github.com/Eya-MN/P2M
echo    Upload manuel des fichiers importants
echo.
echo 🚀 Tentative de solution 1...
pause

git lfs uninstall
echo ✅ LFS désactivé
echo 🔄 Tentative de push...
git push origin master

echo.
echo ✅ Opération terminée
pause
