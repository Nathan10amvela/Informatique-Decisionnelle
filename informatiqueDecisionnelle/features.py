"""
Module pour la préparation et l'ingénierie des caractéristiques
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import os

def prepare_data(train_df, test_df=None, is_single_dataset=False, feature_engineering=True, 
                balance_classes=True, feature_selection=True):
    """
    Prépare les données pour l'entraînement et les tests
    
    Args:
        train_df: DataFrame d'entraînement
        test_df: DataFrame de test (optionnel)
        is_single_dataset: Booléen indiquant si on travaille avec un seul dataset
        feature_engineering: Si True, applique l'ingénierie des caractéristiques
        balance_classes: Si True, équilibre les classes dans les données d'entraînement
        feature_selection: Si True, applique la sélection de caractéristiques
        
    Returns:
        Un tuple contenant les données préparées
    """
    print("\n🔄 Préparation des données pour le machine learning...")
    
    # Séparation des features et de la cible
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    
    if is_single_dataset:
        # Si on utilise un seul dataset, on fait une division train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
    elif test_df is not None:
        # Si on a un dataset de test séparé
        X_test = test_df.drop('target', axis=1)
        y_test = test_df['target']
    else:
        raise ValueError("Erreur: Dataset de test manquant ou paramètre is_single_dataset non spécifié")
    
    # Ingénierie des caractéristiques (si activée)
    if feature_engineering:
        X_train, X_test = apply_feature_engineering(X_train, X_test)
    
    # Équilibrage des classes (si activé)
    if balance_classes:
        X_train, y_train = balance_training_data(X_train, y_train)
    
    # Sélection de caractéristiques (si activée)
    if feature_selection:
        X_train, X_test = select_features(X_train, X_test, y_train)
    
    # Normalisation des caractéristiques
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Conversion en DataFrame pour une meilleure traçabilité 
    # (utile pour l'analyse d'importance des caractéristiques)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    print(f"✅ Dimensions des données après préparation:")
    print(f"- X_train: {X_train.shape}")
    print(f"- X_test: {X_test.shape}")
    print(f"- X_train_scaled: {X_train_scaled.shape}")
    print(f"- X_test_scaled: {X_test_scaled.shape}")
    print(f"- y_train: {y_train.shape}")
    print(f"- y_test: {y_test.shape}")
    
    # Visualisation de la distribution des caractéristiques après normalisation
    os.makedirs('reports/figures', exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(X_train.columns[:min(12, len(X_train.columns))]):
        plt.subplot(3, 4, i+1)
        sns.histplot(X_train_scaled_df[col], kde=True, color='#3498db')
        plt.title(f'Distribution de {col}')
        plt.xlabel('Valeur normalisée')
    
    plt.tight_layout()
    plt.savefig('reports/figures/features_distribution.png')
    plt.close()
    
    return X_train, X_test, X_train_scaled_df, X_test_scaled_df, y_train, y_test, scaler

def apply_feature_engineering(X_train, X_test):
    """
    Applique des techniques d'ingénierie des caractéristiques pour améliorer le modèle
    
    Args:
        X_train: DataFrame des caractéristiques d'entraînement
        X_test: DataFrame des caractéristiques de test
        
    Returns:
        Tuple contenant les DataFrames améliorés
    """
    print("\n✨ Application de l'ingénierie des caractéristiques...")
    
    # Copie pour éviter de modifier les originaux
    X_train_enhanced = X_train.copy()
    X_test_enhanced = X_test.copy()
    
    # 1. Création d'interactions entre les caractéristiques importantes
    print("- Création d'interactions entre caractéristiques...")
    # Interaction entre l'âge et la fréquence cardiaque max
    X_train_enhanced['age_thalach_ratio'] = X_train_enhanced['age'] / X_train_enhanced['thalach']
    X_test_enhanced['age_thalach_ratio'] = X_test_enhanced['age'] / X_test_enhanced['thalach']
    
    # Interaction entre le cholestérol et l'âge
    X_train_enhanced['chol_age_ratio'] = X_train_enhanced['chol'] / X_train_enhanced['age']
    X_test_enhanced['chol_age_ratio'] = X_test_enhanced['chol'] / X_test_enhanced['age']
    
    # 2. Transformation non linéaire de certaines caractéristiques
    print("- Application de transformations non linéaires...")
    # Logarithme du cholestérol (souvent mieux distribué ainsi)
    X_train_enhanced['log_chol'] = np.log1p(X_train_enhanced['chol'])
    X_test_enhanced['log_chol'] = np.log1p(X_test_enhanced['chol'])
    
    # Carré de l'âge (pour capturer les effets non linéaires)
    X_train_enhanced['age_squared'] = X_train_enhanced['age'] ** 2
    X_test_enhanced['age_squared'] = X_test_enhanced['age'] ** 2
    
    # 3. Création d'une caractéristique de risque composite
    print("- Création d'indicateurs de risque composites...")
    # Un score de risque simple basé sur plusieurs facteurs
    risk_factors = ['age', 'chol', 'trestbps']
    
    X_train_enhanced['composite_risk'] = 0
    X_test_enhanced['composite_risk'] = 0
    
    for factor in risk_factors:
        # Normalisation pour chaque facteur
        factor_mean = X_train_enhanced[factor].mean()
        factor_std = X_train_enhanced[factor].std()
        
        X_train_enhanced['composite_risk'] += (X_train_enhanced[factor] - factor_mean) / factor_std
        X_test_enhanced['composite_risk'] += (X_test_enhanced[factor] - factor_mean) / factor_std
    
    # Normalisation finale du score composite
    X_train_enhanced['composite_risk'] = X_train_enhanced['composite_risk'] / len(risk_factors)
    X_test_enhanced['composite_risk'] = X_test_enhanced['composite_risk'] / len(risk_factors)
    
    print(f"✅ Ingénierie des caractéristiques terminée. Nouvelles dimensions: {X_train_enhanced.shape}")
    
    return X_train_enhanced, X_test_enhanced

def balance_training_data(X_train, y_train, method='smote'):
    """
    Équilibre les classes dans les données d'entraînement
    
    Args:
        X_train: DataFrame des caractéristiques d'entraînement
        y_train: Série des étiquettes d'entraînement
        method: Méthode d'équilibrage ('smote', 'undersample', 'oversample')
        
    Returns:
        Tuple contenant les données équilibrées
    """
    print("\n⚖️ Équilibrage des classes dans les données d'entraînement...")
    
    # Comptage initial des classes
    class_counts = pd.Series(y_train).value_counts()
    print(f"Distribution initiale: {dict(class_counts)}")
    
    if method == 'smote':
        # SMOTE: Synthetic Minority Over-sampling Technique
        print("- Utilisation de SMOTE pour créer des exemples synthétiques...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
    elif method == 'undersample':
        # Sous-échantillonnage aléatoire de la classe majoritaire
        print("- Sous-échantillonnage de la classe majoritaire...")
        majority_class = class_counts.idxmax()
        minority_class = class_counts.idxmin()
        
        # Indices des échantillons par classe
        majority_indices = y_train[y_train == majority_class].index
        minority_indices = y_train[y_train == minority_class].index
        
        # Sélection aléatoire d'indices de la classe majoritaire
        np.random.seed(42)
        selected_majority_indices = np.random.choice(
            majority_indices, size=len(minority_indices), replace=False
        )
        
        # Combinaison des indices
        balanced_indices = np.concatenate([minority_indices, selected_majority_indices])
        
        X_resampled = X_train.loc[balanced_indices]
        y_resampled = y_train.loc[balanced_indices]
        
    elif method == 'oversample':
        # Sur-échantillonnage aléatoire de la classe minoritaire
        print("- Sur-échantillonnage de la classe minoritaire...")
        majority_class = class_counts.idxmax()
        minority_class = class_counts.idxmin()
        
        # Indices des échantillons par classe
        majority_indices = np.array(y_train[y_train == majority_class].index)
        minority_indices = np.array(y_train[y_train == minority_class].index)
        
        # Sur-échantillonnage avec remplacement de la classe minoritaire
        np.random.seed(42)
        resampled_minority_indices = np.random.choice(
            minority_indices, size=len(majority_indices), replace=True
        )
        
        # Combinaison des indices
        balanced_indices = np.concatenate([majority_indices, resampled_minority_indices])
        
        X_resampled = X_train.loc[balanced_indices]
        y_resampled = y_train.loc[balanced_indices]
    
    # Comptage final des classes
    new_class_counts = pd.Series(y_resampled).value_counts()
    print(f"Distribution après équilibrage: {dict(new_class_counts)}")
    
    return X_resampled, y_resampled

def select_features(X_train, X_test, y_train, method='combined', k=10):
    """
    Applique une sélection de caractéristiques pour améliorer le modèle
    
    Args:
        X_train: DataFrame des caractéristiques d'entraînement
        X_test: DataFrame des caractéristiques de test
        y_train: Série des étiquettes d'entraînement
        method: Méthode de sélection ('statistical', 'tree_based', 'rfe', 'combined')
        k: Nombre de caractéristiques à sélectionner
        
    Returns:
        Tuple contenant les DataFrames avec les caractéristiques sélectionnées
    """
    print(f"\n🔍 Sélection des caractéristiques les plus importantes (méthode: {method})...")
    
    if method == 'statistical':
        # Sélection basée sur des tests statistiques
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        
        # Création d'un masque pour les caractéristiques sélectionnées
        selected_features_mask = selector.get_support()
        selected_features = X_train.columns[selected_features_mask]
        
    elif method == 'tree_based':
        # Sélection basée sur l'importance des caractéristiques dans un modèle d'arbre
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Tri des caractéristiques par importance
        importances = pd.Series(rf.feature_importances_, index=X_train.columns)
        selected_features = importances.sort_values(ascending=False).head(k).index
        
    elif method == 'rfe':
        # Elimination récursive des caractéristiques
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        selector = RFE(estimator, n_features_to_select=k, step=1)
        selector = selector.fit(X_train, y_train)
        
        # Création d'un masque pour les caractéristiques sélectionnées
        selected_features_mask = selector.support_
        selected_features = X_train.columns[selected_features_mask]
        
    elif method == 'combined':
        # Combinaison de plusieurs méthodes
        print("- Utilisation d'une approche combinée pour la sélection des caractéristiques...")
        
        # Méthode statistique
        stat_selector = SelectKBest(score_func=f_classif, k=k)
        stat_selector.fit(X_train, y_train)
        stat_scores = pd.Series(stat_selector.scores_, index=X_train.columns)
        
        # Méthode basée sur les arbres
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        tree_scores = pd.Series(rf.feature_importances_, index=X_train.columns)
        
        # Normalisation et combinaison des scores
        stat_scores_norm = (stat_scores - stat_scores.min()) / (stat_scores.max() - stat_scores.min())
        tree_scores_norm = (tree_scores - tree_scores.min()) / (tree_scores.max() - tree_scores.min())
        
        combined_scores = (stat_scores_norm + tree_scores_norm) / 2
        selected_features = combined_scores.sort_values(ascending=False).head(k).index
    
    # Application de la sélection aux DataFrames
    X_train_selected = X_train[selected_features].copy()
    X_test_selected = X_test[selected_features].copy()
    
    # Visualisation des caractéristiques sélectionnées
    plt.figure(figsize=(10, 6))
    
    if method in ['statistical', 'combined']:
        scores_to_plot = stat_scores if method == 'statistical' else combined_scores
        sorted_indices = np.argsort(scores_to_plot.values)[::-1]
        plt.barh(range(len(sorted_indices)), scores_to_plot.values[sorted_indices], align='center')
        plt.yticks(range(len(sorted_indices)), scores_to_plot.index[sorted_indices])
        plt.title(f'Scores des caractéristiques ({method})')
        plt.xlabel('Score')
    elif method == 'tree_based':
        importances.sort_values(ascending=False).plot(kind='barh')
        plt.title('Importance des caractéristiques (Random Forest)')
        plt.xlabel('Importance')
    
    plt.tight_layout()
    plt.savefig(f'reports/figures/feature_selection_{method}.png')
    plt.close()
    
    print(f"✅ Sélection terminée. Caractéristiques retenues: {list(selected_features)}")
    
    return X_train_selected, X_test_selected