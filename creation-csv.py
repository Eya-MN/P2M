import os
import pandas as pd

# --- CONFIGURATION ---
DATA_DIR = r'C:\Users\Administrateur\Documents\P2M\dataset_clean'
OUTPUT_CSV = r'C:\Users\Administrateur\Documents\P2M\final_dataset.csv'

def create_master_csv():
    data = []
    
    # On scanne le dossier dataset_clean
    # categories sera la liste : ['armee', 'universite', 'handicap', ...]
    categories = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    for category in categories:
        folder_path = os.path.join(DATA_DIR, category)
        images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_name in images:
            # On stocke le chemin relatif pour que le projet soit portable
            # Exemple : "armee/image_1.jpg"
            relative_path = os.path.join(category, img_name)
            
            data.append({
                "image_path": relative_path,
                "label": category
            })
            
    # Création du DataFrame
    df = pd.DataFrame(data)
    
    # Mélange aléatoire (Shuffle) - Très important pour l'entraînement
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Sauvegarde
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Succès ! {len(df)} images ont été indexées dans {OUTPUT_CSV}")
    print("\n--- Aperçu du CSV ---")
    print(df.head())

if __name__ == "__main__":
    create_master_csv()