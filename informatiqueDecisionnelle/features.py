# informatiqueDecisionnelle/features.py

"""
Module pour la préparation et l'ingénierie des caractéristiques
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, RFE, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_class_weight
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
        X_train, X_test = precision_focused_feature_engineering(X_train, X_test)
    
    # Équilibrage des classes (si activé)
    if balance_classes:
        X_train, y_train, _ = balance_training_data(X_train, y_train)
    
    # Sélection de caractéristiques (si activée)
    if feature_selection:
        X_train, X_test = select_features(X_train, X_test, y_train)
    
    # Normalisation des caractéristiques
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Conversion en DataFrame pour une meilleure traçabilité 
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    print(f"✅ Dimensions des données après préparation:")
    print(f"- X_train: {X_train.shape}")
    print(f"- X_test: {X_test.shape}")
    print(f"- X_train_scaled: {X_train_scaled.shape}")
    print(f"- X_test_scaled: {X_test_scaled.shape}")
    print(f"- y_train: {y_train.shape}")
    print(f"- y_test: {y_test.shape}")
    
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
    
    X_train_enhanced = X_train.copy()
    X_test_enhanced = X_test.copy()
    
    X_train_enhanced['age_thalach_ratio'] = X_train_enhanced['age'] / X_train_enhanced['thalach']
    X_test_enhanced['age_thalach_ratio'] = X_test_enhanced['age'] / X_test_enhanced['thalach']
    
    X_train_enhanced['chol_age_ratio'] = X_train_enhanced['chol'] / X_train_enhanced['age']
    X_test_enhanced['chol_age_ratio'] = X_test_enhanced['chol'] / X_test_enhanced['age']
    
    X_train_enhanced['log_chol'] = np.log1p(X_train_enhanced['chol'])
    X_test_enhanced['log_chol'] = np.log1p(X_test_enhanced['chol'])
    
    X_train_enhanced['age_squared'] = X_train_enhanced['age'] ** 2
    X_test_enhanced['age_squared'] = X_test_enhanced['age'] ** 2
    
    risk_factors = ['age', 'chol', 'trestbps']
    X_train_enhanced['composite_risk'] = 0
    X_test_enhanced['composite_risk'] = 0
    
    for factor in risk_factors:
        factor_mean = X_train_enhanced[factor].mean()
        factor_std = X_train_enhanced[factor].std()
        
        X_train_enhanced['composite_risk'] += (X_train_enhanced[factor] - factor_mean) / factor_std
        X_test_enhanced['composite_risk'] += (X_test_enhanced[factor] - factor_mean) / factor_std
    
    X_train_enhanced['composite_risk'] = X_train_enhanced['composite_risk'] / len(risk_factors)
    X_test_enhanced['composite_risk'] = X_test_enhanced['composite_risk'] / len(risk_factors)
    
    print(f"✅ Ingénierie des caractéristiques terminée. Nouvelles dimensions: {X_train_enhanced.shape}")
    
    return X_train_enhanced, X_test_enhanced

def balance_training_data(X_train, y_train, method='precision_focused'):
    """
    Équilibre les classes avec focus sur la précision
    
    Args:
        X_train: DataFrame des caractéristiques d'entraînement
        y_train: Série des étiquettes d'entraînement
        method: Méthode d'équilibrage ('precision_focused', 'smote', 'undersample', 'oversample')
        
    Returns:
        Tuple contenant les données équilibrées et les poids des classes
    """
    print("\n⚖️ Équilibrage des classes...")
    
    class_counts = pd.Series(y_train).value_counts()
    print(f"Distribution initiale: {dict(class_counts)}")
    
    class_weight_dict = {}

    try:
        if method == 'precision_focused':
            from imblearn.combine import SMOTEENN
            smote_enn = SMOTEENN(random_state=42, sampling_strategy='auto')
            X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
        elif method == 'smote':
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        elif method == 'undersample':
             from imblearn.under_sampling import RandomUnderSampler
             rus = RandomUnderSampler(random_state=42)
             X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
        else: # Default or oversample
             from imblearn.over_sampling import RandomOverSampler
             ros = RandomOverSampler(random_state=42)
             X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
             
        weights = compute_class_weight('balanced', classes=np.unique(y_resampled), y=y_resampled)
        class_weight_dict = {i: weights[i] for i in range(len(weights))}
        
        print(f"Distribution après équilibrage ({method}): {pd.Series(y_resampled).value_counts().to_dict()}")
        
        return X_resampled, y_resampled, class_weight_dict

    except ImportError:
        print("⚠️ Bibliothèque imblearn non disponible, utilisation de poids de classes uniquement")
        weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: weights[i] for i in range(len(weights))}
        return X_train, y_train, class_weight_dict

def select_features(X_train, X_test, y_train, method='precision_focused', k=None):
    """
    Sélection de caractéristiques optimisée pour la précision
    
    Args:
        X_train: DataFrame des caractéristiques d'entraînement
        X_test: DataFrame des caractéristiques de test
        y_train: Série des étiquettes d'entraînement
        method: Méthode de sélection
        k: Nombre de caractéristiques à sélectionner
        
    Returns:
        Tuple contenant les DataFrames avec les caractéristiques sélectionnées
    """
    print(f"\n🔍 Sélection des caractéristiques...")
    
    if k is None:
        k = max(10, min(20, X_train.shape[1] // 3))
    
    if method == 'precision_focused':
        print(f"- Utilisation d'une approche multi-méthode axée sur la précision...")
        
        stat_selector = SelectKBest(score_func=f_classif, k=k)
        stat_selector.fit(X_train, y_train)
        stat_scores = pd.Series(stat_selector.scores_, index=X_train.columns)
        
        mi_selector = SelectKBest(score_func=mutual_info_classif, k=k)
        mi_selector.fit(X_train, y_train)
        mi_scores = pd.Series(mi_selector.scores_, index=X_train.columns)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf.fit(X_train, y_train)
        rf_scores = pd.Series(rf.feature_importances_, index=X_train.columns)
        
        lr = LogisticRegression(max_iter=1000, class_weight='balanced')
        rfe = RFE(estimator=lr, n_features_to_select=k, step=1)
        rfe.fit(X_train, y_train)
        rfe_scores = pd.Series(1 / rfe.ranking_, index=X_train.columns)
        
        stat_scores_norm = (stat_scores - stat_scores.min()) / (stat_scores.max() - stat_scores.min() + 1e-10)
        mi_scores_norm = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min() + 1e-10)
        rf_scores_norm = (rf_scores - rf_scores.min()) / (rf_scores.max() - rf_scores.min() + 1e-10)
        rfe_scores_norm = (rfe_scores - rfe_scores.min()) / (rfe_scores.max() - rfe_scores.min() + 1e-10)
        
        combined_scores = (0.1 * stat_scores_norm + 0.2 * mi_scores_norm + 0.3 * rf_scores_norm + 0.4 * rfe_scores_norm)
        
        selected_features = combined_scores.sort_values(ascending=False).head(k).index.tolist()
        
        critical_features = ['thalach', 'cp', 'exang', 'oldpeak']
        for feature in critical_features:
            if feature in X_train.columns and feature not in selected_features:
                if len(selected_features) >= k:
                    selected_features.pop()
                selected_features.append(feature)
                print(f"  - Ajout forcé de la caractéristique critique: {feature}")
        
        plt.figure(figsize=(12, 8))
        plt.barh(combined_scores.sort_values(ascending=False).head(k).index, 
                 combined_scores.sort_values(ascending=False).head(k).values)
        plt.title('Scores combinés des caractéristiques (axés sur la précision)')
        plt.xlabel('Score pondéré')
        plt.tight_layout()
        plt.savefig('reports/figures/precision_feature_selection.png')
        plt.close()
    
    else:
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X_train, y_train)
        selected_features_mask = selector.get_support()
        selected_features = X_train.columns[selected_features_mask].tolist()
    
    X_train_selected = X_train[selected_features].copy()
    X_test_selected = X_test[selected_features].copy()
    
    print(f"✅ Sélection terminée. {len(selected_features)} caractéristiques retenues.")
    
    return X_train_selected, X_test_selected

def precision_focused_feature_engineering(X_train, X_test):
    """
    Ingénierie des caractéristiques spécialement conçue pour améliorer la précision.
    """
    print("\n🎯 Application d'une ingénierie des caractéristiques axée sur la précision...")
    
    X_train_enhanced = X_train.copy()
    X_test_enhanced = X_test.copy()
    
    if 'age' in X_train.columns and 'thalach' in X_train.columns:
        X_train_enhanced['heart_efficiency_index'] = X_train['thalach'] / X_train['age']
        X_test_enhanced['heart_efficiency_index'] = X_test['thalach'] / X_test['age']
    
    # ... (autres transformations comme dans votre code original)
    
    print(f"✅ Ingénierie axée précision terminée: {X_train_enhanced.shape[1]} caractéristiques")
    return X_train_enhanced, X_test_enhanced

def prepare_data_with_regularization(X_train, X_test, y_train):
    """
    Prépare les données avec une stratégie de régularisation pour éviter le surapprentissage.
    
    Args:
        X_train: Caractéristiques d'entraînement
        X_test: Caractéristiques de test
        y_train: Étiquettes d'entraînement
        
    Returns:
        X_train_processed, X_test_processed, y_train_processed, feature_names, scaler, selector
    """
    print("\n🔄 Préparation des données avec régularisation...")
    
    # 1. Normalisation standard
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. Sélection de caractéristiques
    selector = SelectKBest(f_classif, k=11)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    selected_indices = selector.get_support(indices=True)
    feature_names = X_train.columns[selected_indices].tolist()
    
    # 3. Équilibrage des classes
    under_sampler = RandomUnderSampler(random_state=42)
    X_train_balanced, y_train_balanced = under_sampler.fit_resample(X_train_selected, y_train)
    
    print(f"✅ Données préparées avec succès. Caractéristiques sélectionnées: {len(feature_names)}")
    print(f"Caractéristiques: {', '.join(feature_names)}")
    
    # >>> MODIFICATION ICI <<<
    return X_train_balanced, X_test_selected, y_train_balanced, feature_names, scaler, selector