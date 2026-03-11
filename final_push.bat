@echo off
echo 🔐 AUTHENTIFICATION GITHUB REQUISE
echo ==================================
echo.
echo GitHub demande une authentification dans le navigateur.
echo.
echo 📋 ÉTAPES À SUIVRE:
echo 1️⃣ Un navigateur va s'ouvrir automatiquement
echo 2️⃣ Connectez-vous avec votre compte Eya-MN
echo 3️⃣ Autorisez Git Credential Manager
echo 4️⃣ Revenez ici et appuyez sur une touche
echo.
echo 🔄 Tentative de push automatique...
echo.

git push -u origin master

echo.
echo ✅ Si le push a réussi, le projet est sur GitHub !
echo 🌐 Lien: https://github.com/Eya-MN/P2M
echo.
pause
