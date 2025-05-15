"""
### Projet: Prédiction de Maladies Cardiaques
# Auteur: Wotchoko & Claude
# Date: 15 Avril 2025

Ce script principal coordonne l'ensemble du processus d'analyse et de prédiction
des maladies cardiaques en utilisant plusieurs algorithmes de machine learning.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from informatiqueDecisionnelle.datamanagement import load_datasets
from informatiqueDecisionnelle.dataset import explore_datasets, visualize_data
from informatiqueDecisionnelle.features import prepare_data
from informatiqueDecisionnelle.modeling.train import train_and_evaluate_models, compare_models, optimize_best_model
from informatiqueDecisionnelle.modeling.predict import make_predictions
from informatiqueDecisionnelle.plots import create_report_graphics, create_model_stats, create_test_stats, create_best_model_report

def main():
    """
    Fonction principale qui exécute le workflow complet d'analyse et de prédiction
    """
    print("\n" + "="*80)
    print("🫀 ANALYSE PRÉDICTIVE DE MALADIES CARDIAQUES")
    print("="*80)
    
    # 1. Chargement des données
    print("\n📂 Chargement des datasets...")
    train_df, test_df = load_datasets(
        train_path='data/raw/Heart_disease_statlog.csv',  
        test_path='data/raw/Heart_disease_cleveland_new.csv'
    )
    
    # 2. Exploration et visualisation des données
    train_df = explore_datasets(train_df, "Cleveland (Entraînement)")
    test_df = explore_datasets(test_df, "Statlog (Test)")
    
    visualize_data(train_df, "Dataset Cleveland")
    visualize_data(test_df, "Dataset Statlog")
    
    # 3. Préparation des données
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(
        train_df, test_df, is_single_dataset=False
    )
    
    # 4. Entraînement et évaluation des modèles
    test_results, train_results = train_and_evaluate_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test)
    
    # 5. Comparaison des modèles et identification du meilleur
    best_model_name = compare_models(test_results, X_train.columns)
    
    # 6. Optimisation du meilleur modèle
    optimized_model, optimized_metrics = optimize_best_model(
        best_model_name, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    # 7. Génération des rapports demandés
    # 7.1 PDF avec tous les graphiques importants
    create_report_graphics(train_df, test_df, test_results, optimized_model, X_train.columns)
    
    # 7.2 PDF avec les statistiques de chaque modèle après entraînement
    create_model_stats(train_results, "train")
    
    # 7.3 PDF avec les statistiques de chaque modèle après test
    create_model_stats(test_results, "test")
    
    # 7.4 PDF avec le meilleur modèle et ses courbes
    create_best_model_report(optimized_model, optimized_metrics, X_train.columns, best_model_name)
    
    # 8. Sauvegarde du meilleur modèle
    model_artifacts = {
        'model': optimized_model,
        'scaler': scaler,
        'feature_names': list(X_train.columns),
        'date_created': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model_artifacts, f"models/heart_disease_{best_model_name.lower().replace(' ', '_')}.joblib")
    print(f"\n✅ Modèle sauvegardé sous 'models/heart_disease_{best_model_name.lower().replace(' ', '_')}.joblib'")
    
    print("\n✅ Analyse terminée avec succès!")
    
if __name__ == "__main__":
    main()