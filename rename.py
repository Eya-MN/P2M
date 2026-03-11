import os
import shutil

# --- CONFIGURATION ---
SOURCE_DIR = r'C:\Users\Administrateur\Documents\Dataset'
DEST_DIR = r'C:\Users\Administrateur\Documents\P2M\dataset_clean'

def organiser_dataset():
    # 1. Création du dossier de destination s'il n'existe pas
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
        print(f"📁 Dossier créé : {DEST_DIR}")

    print("🚀 Déplacement et renommage en cours...")

    # 2. Parcours du dossier source (celui de la normalisation)
    for signe in os.listdir(SOURCE_DIR):
        signe_path = os.path.join(SOURCE_DIR, signe)
        
        if os.path.isdir(signe_path):
            # Nettoyage du nom du signe (ex: "armée" -> "armee")
            signe_clean = signe.lower().replace('é', 'e').replace('è', 'e').replace(' ', '_').replace('œ', 'oe')
            
            dest_signe_path = os.path.join(DEST_DIR, signe_clean)
            os.makedirs(dest_signe_path, exist_ok=True)

            for img_name in os.listdir(signe_path):
                source_img = os.path.join(signe_path, img_name)
                
                # Nettoyage du nom du fichier image
                img_name_clean = img_name.lower().replace('é', 'e').replace('è', 'e').replace(' ', '_')
                dest_img = os.path.join(dest_signe_path, img_name_clean)

                # Déplacement (shutil.move déplace, shutil.copy copie)
                # On utilise copy2 ici pour garder une sécurité dans Documents au cas où
                shutil.copy2(source_img, dest_img)

    print(f"✅ Terminé ! Tes images sont prêtes dans : {DEST_DIR}")

if __name__ == "__main__":
    organiser_dataset()