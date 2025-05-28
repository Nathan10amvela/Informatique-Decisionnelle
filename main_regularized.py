"""
Script principal corrigé pour améliorer la cohérence entre entraînement et test
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from informatiqueDecisionnelle.datamanagement import load_datasets
from informatiqueDecisionnelle.dataset import explore_datasets, visualize_data
from informatiqueDecisionnelle.features import prepare_data_with_regularization
from informatiqueDecisionnelle.modeling.train import train_regularized_models, select_best_model_by_generalization
from informatiqueDecisionnelle.plots import create_report_graphics, create_model_stats, create_best_model_report

def main():
    """
    Fonction principale qui exécute le workflow complet d'analyse et de prédiction
    avec des améliorations pour éviter le surapprentissage
    """
    print("\n" + "="*80)
    print(" ANALYSE PRÉDICTIVE DE MALADIES CARDIAQUES (AMÉLIORATION DE LA GÉNÉRALISATION)")
    print("="*80)
    
    # 1. Chargement des données
    print("\n Chargement des données...")
    data_path = 'data/raw/Heart_disease_statlog.csv'
    
    if not os.path.exists(data_path):
        print(f" Fichier de données introuvable: {data_path}")
        return
    
    dataset = pd.read_csv(data_path)
    print(f"Dataset chargé: {len(dataset)} échantillons, {dataset.shape[1]} caractéristiques")
    
    # 2. Division train/test stratifiée et fixe
    X = dataset.drop('target', axis=1)
    y = dataset['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.325, random_state=42, stratify=y
    )
    
    print(f"Division train/test: {len(X_train)} échantillons d'entraînement, {len(X_test)} échantillons de test")
    
    # Reconstruire les DataFrames complets
    train_df = X_train.copy()
    train_df['target'] = y_train.values
    
    test_df = X_test.copy()
    test_df['target'] = y_test.values
    
    # 3. Exploration des données
    train_df = explore_datasets(train_df, "Ensemble d'entraînement (70%)")
    test_df = explore_datasets(test_df, "Ensemble de test (30%)")
    
    visualize_data(train_df, "Ensemble d'entraînement")
    
    # 4. Préparation des données avec stratégie de régularisation améliorée
    X_train_processed, X_test_processed, y_train_processed, feature_names = prepare_data_with_regularization(
        X_train, X_test, y_train
    )
    
    # 5. Entraînement des modèles avec forte régularisation
    test_results, train_results = train_regularized_models(
        X_train_processed, X_test_processed, y_train_processed, y_test, feature_names
    )
    
    # 6. Évaluation et visualisation des résultats
    best_model_name = select_best_model_by_generalization(test_results, train_results, feature_names)
    
    # 7. Création des rapports
    create_report_graphics(train_df, test_df, test_results, 
                          test_results[best_model_name]['model'], feature_names)
    
    create_model_stats(train_results, "train")
    create_model_stats(test_results, "test")
    
    create_best_model_report(
        test_results[best_model_name]['model'], 
        test_results[best_model_name],
        feature_names, 
        best_model_name
    )
    
    # 8. Sauvegarde du meilleur modèle
    model_artifacts = {
        'model': test_results[best_model_name]['model'],
        'scaler': None,  # Le scaler est déjà appliqué dans prepare_data_with_regularization
        'feature_names': feature_names,
        'date_created': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    os.makedirs('models', exist_ok=True)
    model_filename = f"models/heart_disease_{best_model_name.lower().replace(' ', '_')}_regularized.joblib"
    joblib.dump(model_artifacts, model_filename)
    print(f"\n Modèle sauvegardé sous '{model_filename}'")
    
    # 9. Analyse des écarts entre train et test
    print("\n Analyse du surapprentissage:")
    for name, results in test_results.items():
        train_acc = train_results[name]['accuracy']
        test_acc = results['accuracy']
        train_f1 = train_results[name]['f1']
        test_f1 = results['f1']
        
        gap_acc = train_acc - test_acc
        gap_f1 = train_f1 - test_f1
        
        print(f"{name}:")
        print(f"  - Accuracy: {test_acc:.4f} (écart: {gap_acc:.4f})")
        print(f"  - F1-Score: {test_f1:.4f} (écart: {gap_f1:.4f})")
        
        # Modèle correctement régularisé = écart faible
        if gap_acc <= 0.05 and gap_f1 <= 0.05:
            print(f"   Modèle bien régularisé (écart < 5%)")
        elif gap_acc <= 0.1 and gap_f1 <= 0.1:
            print(f"  ⚠️ Régularisation acceptable (écart < 10%)")
        else:
            print(f"   Surapprentissage probable (écart > 10%)")
    
    print("\n✅ Analyse terminée avec succès!")

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
    print("\n Préparation des données avec régularisation...")
    
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
    
    print(f" Données préparées avec succès. Caractéristiques sélectionnées: {len(feature_names)}")
    print(f"Caractéristiques: {', '.join(feature_names)}")
    
    return X_train_balanced, X_test_selected, y_train_balanced, feature_names

if __name__ == "__main__":
    main()