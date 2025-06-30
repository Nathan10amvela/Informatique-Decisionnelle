# main_regularized.py

"""
Script principal corrigé pour améliorer la cohérence entre entraînement et test
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from informatiqueDecisionnelle.datamanagement import load_datasets
from informatiqueDecisionnelle.dataset import explore_datasets, visualize_data
from informatiqueDecisionnelle.plots import create_report_graphics, create_model_stats, create_best_model_report

# >>> MODIFICATION ICI : Import de la fonction depuis le module `features` <<<
from informatiqueDecisionnelle.features import prepare_data_with_regularization

# >>> MODIFICATION ICI : Import des fonctions d'entraînement <<<
from informatiqueDecisionnelle.modeling.train import train_regularized_models, select_best_model_by_generalization

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
    
    train_df = X_train.copy()
    train_df['target'] = y_train.values
    
    test_df = X_test.copy()
    test_df['target'] = y_test.values
    
    # 3. Exploration des données
    explore_datasets(train_df, "Ensemble d'entraînement (70%)")
    explore_datasets(test_df, "Ensemble de test (30%)")
    visualize_data(train_df, "Ensemble d'entraînement")
    
    # 4. Préparation des données avec stratégie de régularisation améliorée
    # >>> MODIFICATION ICI : Mise à jour de l'appel de fonction <<<
    X_train_processed, X_test_processed, y_train_processed, feature_names, scaler, selector = prepare_data_with_regularization(
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
    # >>> MODIFICATION ICI : Sauvegarde des nouveaux artefacts <<<
    model_artifacts = {
        'model': test_results[best_model_name]['model'],
        'scaler': scaler,  # Sauvegarde du scaler
        'selector': selector, # Sauvegarde du sélecteur de features
        'feature_names': feature_names, # Noms des features APRES sélection
        'original_feature_names': X_train.columns.tolist(), # Noms AVANT sélection
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
        
        if abs(gap_acc) <= 0.05 and abs(gap_f1) <= 0.05:
            print(f"   Modèle bien régularisé (écart < 5%)")
        elif abs(gap_acc) <= 0.1 and abs(gap_f1) <= 0.1:
            print(f"  ⚠️ Régularisation acceptable (écart < 10%)")
        else:
            print(f"   Surapprentissage probable (écart > 10%)")
    
    print("\n✅ Analyse terminée avec succès!")

# >>> MODIFICATION ICI : Suppression de la fonction locale redondante <<<
# La fonction prepare_data_with_regularization est maintenant importée, 
# donc la définition locale ci-dessous est supprimée.

if __name__ == "__main__":
    main()