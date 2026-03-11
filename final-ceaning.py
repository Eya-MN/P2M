import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# --- CONFIGURATION ---
CSV_PATH = r'C:\Users\Administrateur\Documents\P2M\final_dataset.csv'
BASE_PATH = r'C:\Users\Administrateur\Documents\P2M\dataset_clean'
IMG_SIZE = 224

def load_and_preprocess_data():
    print("⏳ Chargement du CSV et des images...")
    df = pd.read_csv(CSV_PATH)
    
    X = []
    y = []

    for index, row in df.iterrows():
        img_path = os.path.join(BASE_PATH, row['image_path'])
        
        # 1. Lecture de l'image
        img = cv2.imread(img_path)
        if img is not None:
            # 2. Conversion BGR -> RGB (Ce qu'on a vu juste avant)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 3. Normalisation mathématique (0 à 1)
            img = img.astype('float32') / 255.0
            
            X.append(img)
            y.append(row['label'])

    X = np.array(X)
    print(f"✅ {len(X)} images chargées avec succès.")

    # 4. Encodage des labels (Texte -> Chiffres)
    # Exemple : 'universite' -> 0, 'armee' -> 1
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # 5. One-Hot Encoding (Chiffres -> Vecteurs)
    # Exemple : 0 -> [1, 0, 0], 1 -> [0, 1, 0]
    num_classes = len(np.unique(y_encoded))
    y_final = to_categorical(y_encoded, num_classes=num_classes)

    # 6. Séparation Train (80%) / Test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_final, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"📊 Données d'entraînement : {X_train.shape}")
    print(f"📊 Données de test : {X_test.shape}")
    print(f"🏷️ Classes détectées ({num_classes}) : {label_encoder.classes_}")
    
    return X_train, X_test, y_train, y_test, label_encoder

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, encoder = load_and_preprocess_data()