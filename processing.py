import os
import pandas as pd

# --- CONFIGURATION ---
DATASET_DIR = r'C:\Users\Administrateur\Documents\P2M\dataset'
OUTPUT_CSV = r'C:\Users\Administrateur\Documents\P2M\dataset_rgb.csv'

def generate_csv():
    data = []
    
    # 1. Parcourir les dossiers des signes (université, khedma, etc.)
    for signe in os.listdir(DATASET_DIR):
        signe_path = os.path.join(DATASET_DIR, signe)
        
        if not os.path.isdir(signe_path):
            continue
            
        # 2. Parcourir les dossiers des personnes (yasmine, etc.)
        for personne in os.listdir(signe_path):
            pers_path = os.path.join(signe_path, personne)
            
            if not os.path.isdir(pers_path):
                continue
                
            # 3. Lister les images
            for img_name in os.listdir(pers_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(pers_path, img_name)
                    # On ajoute le chemin et le label (nom du signe)
                    data.append({
                        "path": full_path,
                        "label": signe
                    })

    # 4. Créer le DataFrame et sauvegarder en CSV
    df = pd.DataFrame(data)
    
    # Mélanger les données (optionnel mais recommandé)
    df = df.sample(frac=1).reset_index(drop=True)
    
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"✅ CSV généré avec succès : {len(df)} images trouvées.")
    print(df.head()) # Affiche les 5 premières lignes pour vérifier

if __name__ == "__main__":
    generate_csv()