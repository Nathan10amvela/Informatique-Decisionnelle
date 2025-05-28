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

      # X_train, X_test = apply_feature_engineering(X_train, X_test)
        X_train, X_test = precision_focused_feature_engineering(X_train, X_test)
    
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

# Dans informatiqueDecisionnelle/features.py, remplacez la fonction balance_training_data

def balance_training_data(X_train, y_train, method='precision_focused'):
    """
    Équilibre les classes avec focus sur la précision
    
    Args:
        X_train: DataFrame des caractéristiques d'entraînement
        y_train: Série des étiquettes d'entraînement
        method: Méthode d'équilibrage ('precision_focused', 'smote', 'undersample', 'oversample')
        
    Returns:
        Tuple contenant les données équilibrées
    """
    print("\n⚖️ Équilibrage des classes avec focus sur la précision...")
    
    # Comptage initial des classes
    class_counts = pd.Series(y_train).value_counts()
    print(f"Distribution initiale: {dict(class_counts)}")
    
    if method == 'precision_focused':
        # Approche avancée qui favorise la précision:
        # 1. SMOTE-ENN: SMOTE suivi de nettoyage par Edited Nearest Neighbors
        # 2. Pondération des classes pour l'entraînement
        
        try:
            from imblearn.combine import SMOTEENN
            
            print("- Utilisation de SMOTE-ENN pour équilibrer les classes...")
            smote_enn = SMOTEENN(random_state=42, sampling_strategy='auto')
            X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
            
            # Calculer les poids pour les classificateurs qui supportent class_weight
            from sklearn.utils.class_weight import compute_class_weight
            weights = compute_class_weight('balanced', classes=np.unique(y_resampled), y=y_resampled)
            class_weight_dict = {i: weights[i] for i in range(len(weights))}
            
            # Ajuster légèrement les poids pour favoriser la précision
            # Augmenter le poids de la classe positive pour réduire les faux positifs
            if 1 in class_weight_dict:
                class_weight_dict[1] *= 1.2  # Augmentation de 20% pour la classe positive
                
            # SMOTE-ENN peut parfois créer trop d'exemples synthétiques
            # Si le déséquilibre s'inverse trop, on peut sous-échantillonner légèrement
            new_class_counts = pd.Series(y_resampled).value_counts()
            
            # Si la classe positive devient trop dominante (> 60%), rééquilibrer
            if new_class_counts.get(1, 0) / len(y_resampled) > 0.6:
                from imblearn.under_sampling import RandomUnderSampler
                
                print("  - Ajustement final par sous-échantillonnage...")
                target_ratio = {0: int(len(y_resampled) * 0.45), 1: int(len(y_resampled) * 0.55)}
                rus = RandomUnderSampler(random_state=42, sampling_strategy=target_ratio)
                X_resampled, y_resampled = rus.fit_resample(X_resampled, y_resampled)
            
            print(f"Distribution après équilibrage: {pd.Series(y_resampled).value_counts().to_dict()}")
            print(f"Poids des classes: {class_weight_dict}")
            
            return X_resampled, y_resampled, class_weight_dict
            
        except ImportError:
            print("⚠️ Bibliothèque imblearn non disponible, utilisation de poids de classes uniquement")
            
            # Calculer directement les poids
            from sklearn.utils.class_weight import compute_class_weight
            weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weight_dict = {i: weights[i] for i in range(len(weights))}
            
            # Ajuster pour favoriser la précision
            if 1 in class_weight_dict:
                class_weight_dict[1] *= 1.2
                
            return X_train, y_train, class_weight_dict
            
    elif method == 'smote':
        # SMOTE standard
        from imblearn.over_sampling import SMOTE
        
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        # Calculer les poids
        from sklearn.utils.class_weight import compute_class_weight
        weights = compute_class_weight('balanced', classes=np.unique(y_resampled), y=y_resampled)
        class_weight_dict = {i: weights[i] for i in range(len(weights))}
        
        return X_resampled, y_resampled, class_weight_dict
    
    # Autres méthodes inchangées...
    else:
        # Si méthode non reconnue, utiliser simplement les poids de classes
        from sklearn.utils.class_weight import compute_class_weight
        weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: weights[i] for i in range(len(weights))}
        
        return X_train, y_train, class_weight_dict
    


# Dans informatiqueDecisionnelle/features.py, remplacez la fonction select_features

def select_features(X_train, X_test, y_train, method='precision_focused', k=None):
    """
    Sélection de caractéristiques optimisée pour la précision
    
    Args:
        X_train: DataFrame des caractéristiques d'entraînement
        X_test: DataFrame des caractéristiques de test
        y_train: Série des étiquettes d'entraînement
        method: Méthode de sélection ('precision_focused', 'statistical', 'tree_based', 'rfe', 'combined')
        k: Nombre de caractéristiques à sélectionner (auto-déterminé si None)
        
    Returns:
        Tuple contenant les DataFrames avec les caractéristiques sélectionnées
    """
    print(f"\n🔍 Sélection des caractéristiques axée sur la précision...")
    
    # Déterminer automatiquement le nombre de caractéristiques si non spécifié
    if k is None:
        # Une bonne règle empirique est de garder environ 1/3 des caractéristiques
        # mais au moins 10 et pas plus de 20
        k = max(10, min(20, X_train.shape[1] // 3))
    
    if method == 'precision_focused':
        # Combinaison de méthodes axées sur la réduction des faux positifs
        print(f"- Utilisation d'une approche multi-méthode axée sur la précision...")
        
        from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.feature_selection import RFE
        
        # 1. Méthode statistique (ANOVA F-value)
        stat_selector = SelectKBest(score_func=f_classif, k=k)
        stat_selector.fit(X_train, y_train)
        stat_scores = pd.Series(stat_selector.scores_, index=X_train.columns)
        
        # 2. Information mutuelle
        mi_selector = SelectKBest(score_func=mutual_info_classif, k=k)
        mi_selector.fit(X_train, y_train)
        mi_scores = pd.Series(mi_selector.scores_, index=X_train.columns)
        
        # 3. Importance basée sur Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf.fit(X_train, y_train)
        rf_scores = pd.Series(rf.feature_importances_, index=X_train.columns)
        
        # 4. Élimination récursive des caractéristiques avec régression logistique
        # (Plus stable pour la précision)
        lr = LogisticRegression(max_iter=1000, class_weight='balanced')
        rfe = RFE(estimator=lr, n_features_to_select=k, step=1)
        rfe.fit(X_train, y_train)
        rfe_scores = pd.Series(rfe.ranking_, index=X_train.columns)
        # Inverser les scores RFE car les rangs plus bas sont meilleurs
        rfe_scores = 1 / rfe_scores
        
        # Pondération des scores des différentes méthodes pour favoriser la précision
        # Random Forest et RFE avec logreg sont généralement meilleurs pour la précision
        stat_scores_norm = (stat_scores - stat_scores.min()) / (stat_scores.max() - stat_scores.min() + 1e-10)
        mi_scores_norm = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min() + 1e-10)
        rf_scores_norm = (rf_scores - rf_scores.min()) / (rf_scores.max() - rf_scores.min() + 1e-10)
        rfe_scores_norm = (rfe_scores - rfe_scores.min()) / (rfe_scores.max() - rfe_scores.min() + 1e-10)
        
        # Pondération favorisant la précision
        combined_scores = (
            0.1 * stat_scores_norm + 
            0.2 * mi_scores_norm + 
            0.3 * rf_scores_norm + 
            0.4 * rfe_scores_norm
        )
        
        # Sélection des meilleures caractéristiques
        selected_features = combined_scores.sort_values(ascending=False).head(k).index.tolist()
        
        # Vérification de l'inclusion de caractéristiques cruciales
        critical_features = ['thalach', 'cp', 'exang', 'oldpeak']
        for feature in critical_features:
            if feature in X_train.columns and feature not in selected_features:
                # Ajouter les caractéristiques critiques et retirer la moins importante
                if len(selected_features) >= k:
                    selected_features.pop()
                selected_features.append(feature)
                print(f"  - Ajout forcé de la caractéristique critique: {feature}")
        
        # Créer des visualisations des scores
        plt.figure(figsize=(12, 8))
        plt.barh(combined_scores.sort_values(ascending=False).head(k).index, 
                combined_scores.sort_values(ascending=False).head(k).values)
        plt.title('Scores combinés des caractéristiques (axés sur la précision)')
        plt.xlabel('Score pondéré')
        plt.tight_layout()
        plt.savefig('reports/figures/precision_feature_selection.png')
        plt.close()
    
    else:
        # Utiliser les méthodes existantes si méthode non spécifiée
        from sklearn.feature_selection import SelectKBest, f_classif
        
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X_train, y_train)
        
        selected_features_mask = selector.get_support()
        selected_features = X_train.columns[selected_features_mask].tolist()
    
    # Application de la sélection aux DataFrames
    X_train_selected = X_train[selected_features].copy()
    X_test_selected = X_test[selected_features].copy()
    
    print(f"✅ Sélection terminée. {len(selected_features)} caractéristiques retenues:")
    print(f"   {', '.join(selected_features[:5])}" + (f"... et {len(selected_features)-5} autres" if len(selected_features) > 5 else ""))
    
    return X_train_selected, X_test_selected







# Ajoutez cette fonction à la fin du fichier informatiqueDecisionnelle/features.py

def precision_focused_feature_engineering(X_train, X_test):
    """
    Ingénierie des caractéristiques spécialement conçue pour améliorer la précision
    et réduire les faux positifs.
    
    Args:
        X_train: DataFrame d'entraînement
        X_test: DataFrame de test
        
    Returns:
        X_train_enhanced, X_test_enhanced: DataFrames avec caractéristiques améliorées
    """
    print("\n🎯 Application d'une ingénierie des caractéristiques axée sur la précision...")
    
    # Copie pour éviter de modifier les originaux
    X_train_enhanced = X_train.copy()
    X_test_enhanced = X_test.copy()
    
    # 1. Caractéristiques cliniquement significatives
    
    # Indice de masse corporelle (BMI proxy)
    if 'weight' in X_train.columns and 'height' in X_train.columns:
        X_train_enhanced['bmi'] = X_train['weight'] / (X_train['height']**2)
        X_test_enhanced['bmi'] = X_test['weight'] / (X_test['height']**2)
    
    # Indice de risque Framingham simplifié
    risk_factors = []
    
    # Âge comme facteur de risque non-linéaire (risque exponentiel avec l'âge)
    if 'age' in X_train.columns:
        X_train_enhanced['age_risk'] = np.exp((X_train['age'] - 40) / 10) / 10
        X_test_enhanced['age_risk'] = np.exp((X_test['age'] - 40) / 10) / 10
        risk_factors.append('age_risk')
    
    # Cholestérol comme facteur de risque par paliers
    if 'chol' in X_train.columns:
        X_train_enhanced['chol_risk'] = pd.cut(
            X_train['chol'], 
            bins=[0, 200, 240, 1000], 
            labels=[0, 1, 2]
        ).astype(float)
        X_test_enhanced['chol_risk'] = pd.cut(
            X_test['chol'], 
            bins=[0, 200, 240, 1000], 
            labels=[0, 1, 2]
        ).astype(float)
        risk_factors.append('chol_risk')
    
    # Pression artérielle comme facteur de risque par paliers
    if 'trestbps' in X_train.columns:
        X_train_enhanced['bp_risk'] = pd.cut(
            X_train['trestbps'], 
            bins=[0, 120, 140, 500], 
            labels=[0, 1, 2]
        ).astype(float)
        X_test_enhanced['bp_risk'] = pd.cut(
            X_test['trestbps'], 
            bins=[0, 120, 140, 500], 
            labels=[0, 1, 2]
        ).astype(float)
        risk_factors.append('bp_risk')
    
    # Score composite pondéré médicalement (basé sur les guidelines cliniques)
    if len(risk_factors) > 0:
        X_train_enhanced['clinical_risk_score'] = 0
        X_test_enhanced['clinical_risk_score'] = 0
        
        # Poids différents selon l'importance médicale
        weights = {'age_risk': 3, 'chol_risk': 2, 'bp_risk': 2}
        
        for factor in risk_factors:
            weight = weights.get(factor, 1)
            X_train_enhanced['clinical_risk_score'] += X_train_enhanced[factor] * weight
            X_test_enhanced['clinical_risk_score'] += X_test_enhanced[factor] * weight
    
    # 2. Interactions spécifiques à haute valeur discriminante
    
    # Interaction âge et rythme cardiaque maximum (index significatif)
    if 'age' in X_train.columns and 'thalach' in X_train.columns:
        # Cette caractéristique est un fort indicateur physiologique du risque cardiaque
        X_train_enhanced['heart_efficiency_index'] = X_train['thalach'] / X_train['age']
        X_test_enhanced['heart_efficiency_index'] = X_test['thalach'] / X_test['age']
        
        # Classification de cet index en catégories à risque
        X_train_enhanced['heart_efficiency_category'] = pd.qcut(
            X_train_enhanced['heart_efficiency_index'], 
            q=4, 
            labels=False
        )
        X_test_enhanced['heart_efficiency_category'] = pd.qcut(
            X_test_enhanced['heart_efficiency_index'], 
            q=4, 
            labels=False,
            duplicates='drop'  # En cas de valeurs dupliquées aux limites des quantiles
        )
    
    # 3. Indicateurs spécifiques de douleur thoracique
    
    if 'cp' in X_train.columns:
        # Transformation one-hot améliorée de la douleur thoracique
        # cp=1 (angine typique) est le plus indicatif
        for cp_type in range(4):
            X_train_enhanced[f'cp_type_{cp_type}'] = (X_train['cp'] == cp_type).astype(int)
            X_test_enhanced[f'cp_type_{cp_type}'] = (X_test['cp'] == cp_type).astype(int)
            
        # Combinaison de douleur thoracique et d'autres symptômes
        if 'exang' in X_train.columns:
            # Douleur thoracique + angine induite par l'exercice est un très fort indicateur
            X_train_enhanced['angina_severity'] = X_train['exang'] * (3 - X_train['cp'])
            X_test_enhanced['angina_severity'] = X_test['exang'] * (3 - X_test['cp'])
    
    # 4. Facteurs de risque combinés
    
    risk_columns = []
    
    if 'sex' in X_train.columns:
        risk_columns.append('sex')  # 1 pour homme = risque plus élevé
    
    if 'fbs' in X_train.columns:
        risk_columns.append('fbs')  # Glycémie élevée
    
    if 'exang' in X_train.columns:
        risk_columns.append('exang')  # Angine liée à l'exercice
    
    if len(risk_columns) > 1:
        # Compte des facteurs de risque présents
        X_train_enhanced['risk_factor_count'] = X_train[risk_columns].sum(axis=1)
        X_test_enhanced['risk_factor_count'] = X_test[risk_columns].sum(axis=1)
    
    # 5. Analyse spectrale des caractéristiques numériques clés
    
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    available_numeric = [col for col in numeric_cols if col in X_train.columns]
    
    if len(available_numeric) >= 3:
        from sklearn.decomposition import PCA
        
        # Normalisation locale pour PCA
        from sklearn.preprocessing import StandardScaler
        scaler_pca = StandardScaler()
        X_train_num_scaled = scaler_pca.fit_transform(X_train[available_numeric])
        X_test_num_scaled = scaler_pca.transform(X_test[available_numeric])
        
        # Appliquer PCA pour extraire les composantes principales
        pca = PCA(n_components=min(3, len(available_numeric)))
        X_train_pca = pca.fit_transform(X_train_num_scaled)
        X_test_pca = pca.transform(X_test_num_scaled)
        
        # Ajouter les composantes comme nouvelles caractéristiques
        for i in range(X_train_pca.shape[1]):
            X_train_enhanced[f'pca_comp_{i+1}'] = X_train_pca[:, i]
            X_test_enhanced[f'pca_comp_{i+1}'] = X_test_pca[:, i]
            
        print(f"Composantes PCA expliquées: {pca.explained_variance_ratio_}")
    
    print(f"✅ Ingénierie axée précision terminée: {X_train_enhanced.shape[1]} caractéristiques")
    return X_train_enhanced, X_test_enhanced


def apply_basic_feature_engineering(X_train, X_test):
    """
    Applique une ingénierie des caractéristiques basique et robuste
    qui est moins susceptible de causer du surapprentissage
    
    Args:
        X_train: DataFrame des caractéristiques d'entraînement
        X_test: DataFrame des caractéristiques de test
        
    Returns:
        Tuple des DataFrames avec caractéristiques améliorées
    """
    print("- Application d'une ingénierie des caractéristiques modérée...")
    
    # Copie pour éviter de modifier les originaux
    X_train_enhanced = X_train.copy()
    X_test_enhanced = X_test.copy()
    
    # 1. Simples ratios physiologiques pertinents médicalement
    if 'age' in X_train.columns and 'thalach' in X_train.columns:
        # Ratio entre fréquence cardiaque max et âge (indicateur connu)
        X_train_enhanced['age_thalach_ratio'] = X_train_enhanced['thalach'] / X_train_enhanced['age']
        X_test_enhanced['age_thalach_ratio'] = X_test_enhanced['thalach'] / X_test_enhanced['age']
    
    if 'trestbps' in X_train.columns and 'thalach' in X_train.columns:
        # Ratio pression artérielle / rythme cardiaque (indice d'effort cardiaque)
        X_train_enhanced['bp_thalach_ratio'] = X_train_enhanced['trestbps'] / X_train_enhanced['thalach']
        X_test_enhanced['bp_thalach_ratio'] = X_test_enhanced['trestbps'] / X_test_enhanced['thalach']
    
    # Pas de transformation polynomiale ou d'interactions complexes
    # Pas de création d'un grand nombre de caractéristiques
    
    print(f"✅ Ingénierie modérée terminée. Nombre final de caractéristiques: {X_train_enhanced.shape[1]}")
    
    return X_train_enhanced, X_test_enhanced

def select_features_simple(X_train, X_test, y_train, max_features=10):
    """
    Sélection de caractéristiques simple et robuste
    
    Args:
        X_train, X_test: DataFrames normalisés
        y_train: Labels d'entraînement
        max_features: Nombre maximum de caractéristiques à sélectionner
        
    Returns:
        X_train_selected, X_test_selected
    """
    print(f"- Sélection des {max_features} caractéristiques les plus pertinentes...")
    
    # Utiliser un simple test ANOVA F-value
    selector = SelectKBest(score_func=f_classif, k=min(max_features, X_train.shape[1]))
    selector.fit(X_train, y_train)  # Fit uniquement sur les données d'entraînement
    
    # Récupérer les indices et scores des caractéristiques sélectionnées
    selected_indices = selector.get_support(indices=True)
    selected_features = X_train.columns[selected_indices]
    feature_scores = selector.scores_[selected_indices]
    
    # Sélectionner les caractéristiques dans les DataFrames
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    # Afficher les caractéristiques sélectionnées et leurs scores
    print("Caractéristiques sélectionnées:")
    for feature, score in zip(selected_features, feature_scores):
        print(f"  - {feature}: {score:.4f}")
    
    return X_train_selected, X_test_selected

def balanced_sampling(X_train, y_train):
    """
    Équilibre les classes en utilisant une technique de sous-échantillonnage
    qui est moins susceptible de créer du surapprentissage que SMOTE
    
    Args:
        X_train: DataFrame des caractéristiques
        y_train: Série des étiquettes
        
    Returns:
        X_balanced, y_balanced: Données équilibrées
    """
    from imblearn.under_sampling import RandomUnderSampler
    
    print("- Équilibrage des classes par sous-échantillonnage aléatoire...")
    
    # Compter les classes avant
    class_counts_before = pd.Series(y_train).value_counts()
    print(f"  Distribution avant: {dict(class_counts_before)}")
    
    # Utiliser un sous-échantillonnage aléatoire (moins risqué que la sur-génération)
    sampler = RandomUnderSampler(random_state=42)
    X_balanced, y_balanced = sampler.fit_resample(X_train, y_train)
    
    # Compter les classes après
    class_counts_after = pd.Series(y_balanced).value_counts()
    print(f"  Distribution après: {dict(class_counts_after)}")
    
    return X_balanced, y_balanced

def prepare_data_with_regularization(X_train, X_test, y_train):
    """
    Prépare les données avec une stratégie de régularisation pour éviter le surapprentissage
    
    Args:
        X_train: Caractéristiques d'entraînement
        X_test: Caractéristiques de test
        y_train: Étiquettes d'entraînement
        
    Returns:
        X_train_processed, X_test_processed, y_train_processed, feature_names
    """
    print("\n🔄 Préparation des données avec régularisation...")
    
    # 1. Application de transformations simples et robustes
    # (Nous évitons les transformations complexes susceptibles de surajuster)
    
    # Normalisation standard - IMPORTANT: fit sur train, transform sur test
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. Sélection de caractéristiques statistiquement significatives
    # Utiliser une méthode robuste comme ANOVA
    from sklearn.feature_selection import SelectKBest, f_classif
    
    # Sélectionner moins de caractéristiques pour réduire le surapprentissage
    # Pour un petit dataset, 8-10 caractéristiques peuvent suffire
    selector = SelectKBest(f_classif, k=10)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    # Récupérer les noms des caractéristiques sélectionnées
    selected_indices = selector.get_support(indices=True)
    feature_names = [X_train.columns[i] for i in selected_indices]
    
    # 3. Équilibrage des classes avec prudence
    # Pour un petit dataset, on utilise une technique non agressive
    from imblearn.under_sampling import RandomUnderSampler
    
    # Sous-échantillonnage aléatoire est moins susceptible de créer des points artificiels
    # qui ne généralisent pas bien (comme SMOTE peut le faire)
    under_sampler = RandomUnderSampler(random_state=42)
    X_train_balanced, y_train_balanced = under_sampler.fit_resample(X_train_selected, y_train)
    
    print(f"✅ Données préparées avec succès. Caractéristiques sélectionnées: {len(feature_names)}")
    print(f"Caractéristiques: {', '.join(feature_names)}")
    
    return X_train_balanced, X_test_selected, y_train_balanced, feature_names