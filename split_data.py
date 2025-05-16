# split_data.py
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def split_and_save_data(input_file, train_ratio=0.65, random_state=42):
    """
    Divise un dataset en ensembles d'entraînement et de test,
    puis sauvegarde les résultats dans des fichiers séparés.
    
    Args:
        input_file: Chemin du fichier CSV d'entrée
        train_ratio: Proportion des données à utiliser pour l'entraînement (par défaut: 0.65)
        random_state: Graine pour la reproductibilité
    
    Returns:
        Chemins des fichiers d'entraînement et de test créés
    """
    print(f"Chargement du dataset depuis {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"Dataset original: {df.shape[0]} échantillons, {df.shape[1]} caractéristiques")
    
    # Vérification de la distribution des classes
    class_counts = df['target'].value_counts()
    print("\nDistribution des classes dans le dataset original:")
    for label, count in class_counts.items():
        print(f"- Classe {label}: {count} échantillons ({count/len(df)*100:.2f}%)")
    
    # Division en ensembles d'entraînement et de test avec stratification
    train_df, test_df = train_test_split(
        df, 
        test_size=(1-train_ratio),
        random_state=random_state,
        stratify=df['target']  # Maintient les proportions de classe
    )
    
    # Vérification des tailles
    print(f"\nEnsemble d'entraînement: {train_df.shape[0]} échantillons ({train_df.shape[0]/df.shape[0]*100:.2f}%)")
    print(f"Ensemble de test: {test_df.shape[0]} échantillons ({test_df.shape[0]/df.shape[0]*100:.2f}%)")
    
    # Vérification de la distribution des classes après division
    print("\nDistribution des classes après division:")
    print("Ensemble d'entraînement:")
    train_class_counts = train_df['target'].value_counts()
    for label, count in train_class_counts.items():
        print(f"- Classe {label}: {count} échantillons ({count/len(train_df)*100:.2f}%)")
    
    print("\nEnsemble de test:")
    test_class_counts = test_df['target'].value_counts()
    for label, count in test_class_counts.items():
        print(f"- Classe {label}: {count} échantillons ({count/len(test_df)*100:.2f}%)")
    
    # Création du dossier de sortie si nécessaire
    output_dir = os.path.dirname(input_file)
    if not output_dir:
        output_dir = '.'
    
    # Détermination des noms de fichiers de sortie
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    train_file = f"{output_dir}/{base_name}_train_65pct.csv"
    test_file = f"{output_dir}/{base_name}_test_35pct.csv"
    
    # Sauvegarde des ensembles
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"\nEnsemble d'entraînement sauvegardé dans: {train_file}")
    print(f"Ensemble de test sauvegardé dans: {test_file}")
    
    return train_file, test_file

if __name__ == "__main__":
    # Utilisation du script en standalone
    input_file = "data/raw/Heart_disease_cleveland_new.csv"
    train_file, test_file = split_and_save_data(input_file, train_ratio=0.7)