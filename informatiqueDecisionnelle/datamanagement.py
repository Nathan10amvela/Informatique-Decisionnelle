"""
Module de gestion des données pour le projet de prédiction de maladies cardiaques.
"""

import pandas as pd
import numpy as np

# Modification de la fonction load_datasets dans informatiqueDecisionnelle/datamanagement.py

def load_datasets(train_path, test_path):
    """
    Charge les datasets d'entraînement et de test
    
    Args:
        train_path: Chemin vers le dataset d'entraînement
        test_path: Chemin vers le dataset de test
        
    Returns:
        Tuple contenant les DataFrames d'entraînement et de test
    """
    print("\n📂 Chargement des datasets...")
    
    try:
        # Dataset d'entraînement
        train_df = pd.read_csv(train_path)
        print(f"✅ Dataset d'entraînement chargé avec succès: {train_path}")
        print(f"   {len(train_df)} échantillons, {train_df.shape[1]} caractéristiques")
        
        # Dataset de test
        test_df = pd.read_csv(test_path)
        print(f"✅ Dataset de test chargé avec succès: {test_path}")
        print(f"   {len(test_df)} échantillons, {test_df.shape[1]} caractéristiques")
        
    except FileNotFoundError as e:
        print(f"❌ Erreur: Fichier introuvable - {e}")
        return None, None
    
    # Vérification rapide de la distribution des classes
    print("\nDistribution des classes:")
    print("Ensemble d'entraînement:")
    print(train_df['target'].value_counts())
    print("\nEnsemble de test:")
    print(test_df['target'].value_counts())
    
    return train_df, test_df

def check_class_balance(df, target_col='target'):
    """
    Vérifie l'équilibre des classes dans le dataset
    
    Args:
        df: DataFrame à analyser
        target_col: Nom de la colonne cible
        
    Returns:
        DataFrame contenant des informations sur l'équilibre des classes
    """
    class_counts = df[target_col].value_counts()
    class_props = df[target_col].value_counts(normalize=True) * 100
    
    balance_df = pd.DataFrame({
        'Count': class_counts,
        'Percentage': class_props
    })
    
    print("\n📊 Équilibre des classes:")
    print(balance_df)
    
    # Calcul du ratio d'équilibre (plus proche de 1 est mieux)
    min_class = class_counts.min()
    max_class = class_counts.max()
    balance_ratio = min_class / max_class
    
    print(f"Ratio d'équilibre: {balance_ratio:.4f} (1.0 = parfaitement équilibré)")
    
    if balance_ratio < 0.8:
        print("⚠️ Les classes sont déséquilibrées. Des techniques d'équilibrage pourraient être nécessaires.")
    
    return balance_df

def handle_missing_values(df, strategy='median'):
    """
    Gère les valeurs manquantes dans le DataFrame
    
    Args:
        df: DataFrame à traiter
        strategy: Stratégie à utiliser ('median', 'mean', 'mode', 'drop')
        
    Returns:
        DataFrame sans valeurs manquantes
    """
    missing_count = df.isnull().sum()
    
    if missing_count.sum() == 0:
        print("✅ Aucune valeur manquante détectée")
        return df
    
    print("\n⚠️ Valeurs manquantes détectées:")
    for col in missing_count[missing_count > 0].index:
        print(f"- {col}: {missing_count[col]} valeurs ({missing_count[col]/len(df)*100:.2f}%)")
    
    # Traitement des valeurs manquantes
    if strategy == 'median':
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
                
        for col in df.select_dtypes(exclude=[np.number]).columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])
                
    elif strategy == 'mean':
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mean())
                
        for col in df.select_dtypes(exclude=[np.number]).columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])
                
    elif strategy == 'mode':
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])
                
    elif strategy == 'drop':
        df = df.dropna()
        print(f"✅ Lignes avec valeurs manquantes supprimées. Nouvelle taille: {len(df)}")
    
    print(f"✅ Valeurs manquantes traitées avec la stratégie '{strategy}'")
    return df

def identify_outliers(df, method='iqr', threshold=1.5):
    """
    Identifie les valeurs aberrantes dans les données numériques
    
    Args:
        df: DataFrame à analyser
        method: Méthode de détection ('iqr' ou 'zscore')
        threshold: Seuil pour la détection
        
    Returns:
        DataFrame indiquant le nombre d'outliers par colonne
    """
    outliers_count = {}
    
    for col in df.select_dtypes(include=[np.number]).columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            outliers_count[col] = len(outliers)
            
        elif method == 'zscore':
            mean = df[col].mean()
            std = df[col].std()
            
            z_scores = abs((df[col] - mean) / std)
            outliers = df[z_scores > threshold][col]
            outliers_count[col] = len(outliers)
    
    outliers_df = pd.DataFrame({
        'Column': outliers_count.keys(),
        'Outliers Count': outliers_count.values(),
        'Percentage': [count/len(df)*100 for count in outliers_count.values()]
    })
    
    print("\n🔍 Détection des valeurs aberrantes:")
    print(outliers_df.sort_values('Outliers Count', ascending=False))
    
    return outliers_df