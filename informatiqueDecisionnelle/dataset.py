import pandas as pd
import numpy as np
from informatiqueDecisionnelle.plots import plot_data_exploration

def explore_dataset(df, title):
    """Réalise une exploration complète du dataset"""
    print(f"\n{'='*80}")
    print(f"EXPLORATION DU DATASET: {title}")
    print(f"{'='*80}")
    
    # Aperçu des données
    print("\n📊 Aperçu des données:")
    print(df.head())
    
    # Informations générales
    print("\n📋 Informations sur le dataframe:")
    print(f"- Nombre de lignes: {df.shape[0]}")
    print(f"- Nombre de colonnes: {df.shape[1]}")
    print(f"- Types de données:")
    for dtype in df.dtypes.value_counts().index:
        n_cols = df.dtypes.value_counts()[dtype]
        print(f"  • {dtype}: {n_cols} colonnes")
    
    # Valeurs manquantes
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("\n⚠️ Valeurs manquantes détectées:")
        for col in missing_values[missing_values > 0].index:
            print(f"- {col}: {missing_values[col]} valeurs manquantes ({missing_values[col]/len(df)*100:.2f}%)")
    else:
        print("\n✅ Aucune valeur manquante détectée")
    
    # Statistiques descriptives
    print("\n📈 Statistiques descriptives:")
    print(df.describe().T)
    
    # Distribution de la variable cible
    if 'target' in df.columns:
        target_counts = df['target'].value_counts()
        print("\n🎯 Distribution de la variable cible:")
        for label, count in target_counts.items():
            percentage = count / len(df) * 100
            print(f"- Classe {label}: {count} échantillons ({percentage:.2f}%)")
    
    # Visualisation des données
    plot_data_exploration(df, title)
    
    return df