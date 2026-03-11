@echo off
echo 🔐 CONFIGURATION GITHUB PERSONAL ACCESS TOKEN
echo ==========================================
echo.
echo Le problème est que GitHub n'accepte pas le push avec le username "Eya-Manaa"
echo Il faut utiliser un Personal Access Token (PAT)
echo.
echo ÉTAPES:
echo 1️⃣ Allez sur https://github.com/settings/tokens
echo 2️⃣ Cliquez sur "Generate new token (classic)"
echo 3️⃣ Cochez "repo" (full control)
echo 4️⃣ Copiez le token généré
echo.
echo ⚠️  NE PARTAGEZ JAMAIS CE TOKEN !
echo.

set /p token="Entrez votre Personal Access Token: "

echo.
echo 🔄 Configuration du remote avec token...
git remote set-url origin https://%token%@github.com/Eya-MN/P2M.git

echo ✅ Configuration terminée
echo 🚀 Tentative de push...
git push -u origin master

echo.
echo ✅ Opération terminée !
pause
