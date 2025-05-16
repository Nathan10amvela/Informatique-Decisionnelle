# main.py modifié
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
    avec une division correcte des données.
    """
    print("\n" + "="*80)
    print("🫀 ANALYSE PRÉDICTIVE DE MALADIES CARDIAQUES (MÉTHODE CORRIGÉE)")
    print("="*80)
    
    # 1. Chargement des datasets correctement séparés
    print("\n📂 Chargement des datasets...")
    train_path = 'data/raw/Heart_disease_statlog_new_train_65pct.csv'
    test_path = 'data/raw/Heart_disease_statlog_new_test_35pct.csv'
    
    # Vérifier si les fichiers existent déjà, sinon les créer
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print("Fichiers séparés non trouvés. Division du dataset original...")
        from split_data import split_and_save_data
        
        train_path, test_path = split_and_save_data(
            'data/raw/Heart_disease_statlog.csv', 
            train_ratio=0.7
        )
    
    # Chargement des datasets séparés
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Ensemble d'entraînement: {len(train_df)} échantillons")
    print(f"Ensemble de test: {len(test_df)} échantillons")
    
    # 2. Exploration et visualisation des données
    train_df = explore_datasets(train_df, "Ensemble d'entraînement (65%)")
    test_df = explore_datasets(test_df, "Ensemble de test (35%)")
    
    visualize_data(train_df, "Ensemble d'entraînement")
    visualize_data(test_df, "Ensemble de test")
    
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
    model_filename = f"models/heart_disease_{best_model_name.lower().replace(' ', '_')}_corrected.joblib"
    joblib.dump(model_artifacts, model_filename)
    print(f"\n✅ Modèle sauvegardé sous '{model_filename}'")
    
    print("\n✅ Analyse terminée avec succès!")

if __name__ == "__main__":
    main()