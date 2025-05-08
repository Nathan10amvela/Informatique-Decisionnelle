from informatiqueDecisionnelle.datamanagement import load_datasets
from informatiqueDecisionnelle.dataset import explore_dataset
from informatiqueDecisionnelle.features import prepare_data
from informatiqueDecisionnelle.modeling.train import (
    train_and_evaluate_models, 
    optimize_model,
    save_model
)

def main():
    """Workflow complet d'analyse prédictive"""
    # 1. Chargement des données
    cleveland_df, statlog_df = load_datasets()
    
    # 2. Exploration des données
    explore_dataset(cleveland_df, "Cleveland (Entraînement)")
    explore_dataset(statlog_df, "Statlog (Test)")
    
    # 3. Préparation des données
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(
        cleveland_df, statlog_df, is_single_dataset=False
    )
    
    # 4. Entraînement et évaluation des modèles
    results, best_model_name = train_and_evaluate_models(
        X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    # 5. Optimisation du meilleur modèle
    optimized_model = optimize_model(
        best_model_name, X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    # 6. Sauvegarde du modèle final
    save_model(optimized_model, scaler, X_train.columns, f"heart_disease_{best_model_name.lower().replace(' ', '_')}")

if __name__ == "__main__":
    main()