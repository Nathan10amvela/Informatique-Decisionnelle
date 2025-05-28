# # main.py modifié
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import joblib
# from informatiqueDecisionnelle.datamanagement import load_datasets
# from informatiqueDecisionnelle.dataset import explore_datasets, visualize_data
# from informatiqueDecisionnelle.features import prepare_data
# from informatiqueDecisionnelle.modeling.train import train_and_evaluate_models, compare_models, optimize_best_model
# from informatiqueDecisionnelle.modeling.predict import make_predictions
# from informatiqueDecisionnelle.plots import create_report_graphics, create_model_stats, create_test_stats, create_best_model_report

# def main():
#     """
#     Fonction principale qui exécute le workflow complet d'analyse et de prédiction
#     avec une division correcte des données.
#     """
#     print("\n" + "="*80)
#     print("🫀 ANALYSE PRÉDICTIVE DE MALADIES CARDIAQUES (MÉTHODE CORRIGÉE)")
#     print("="*80)
    
#     # 1. Chargement des datasets correctement séparés
#     print("\n📂 Chargement des datasets...")
#     train_path = 'data/raw/Heart_disease_statlog_new_train_65pct.csv'
#     test_path = 'data/raw/Heart_disease_statlog_new_test_35pct.csv'
    
#     # Vérifier si les fichiers existent déjà, sinon les créer
#     if not (os.path.exists(train_path) and os.path.exists(test_path)):
#         print("Fichiers séparés non trouvés. Division du dataset original...")
#         from split_data import split_and_save_data
        
#         train_path, test_path = split_and_save_data(
#             'data/raw/Heart_disease_statlog.csv', 
#             train_ratio=0.7
#         )
    
#     # Chargement des datasets séparés
#     train_df = pd.read_csv(train_path)
#     test_df = pd.read_csv(test_path)
    
#     print(f"Ensemble d'entraînement: {len(train_df)} échantillons")
#     print(f"Ensemble de test: {len(test_df)} échantillons")
    
#     # 2. Exploration et visualisation des données
#     train_df = explore_datasets(train_df, "Ensemble d'entraînement (65%)")
#     test_df = explore_datasets(test_df, "Ensemble de test (35%)")
    
#     visualize_data(train_df, "Ensemble d'entraînement")
#     visualize_data(test_df, "Ensemble de test")
    
#     # 3. Préparation des données
#     X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(
#         train_df, test_df, is_single_dataset=False
#     )
    
#     # 4. Entraînement et évaluation des modèles
#     test_results, train_results = train_and_evaluate_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test)
    
#     # 5. Comparaison des modèles et identification du meilleur
#     best_model_name = compare_models(test_results, X_train.columns)
    
#     # 6. Optimisation du meilleur modèle
#     optimized_model, optimized_metrics = optimize_best_model(
#         best_model_name, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test
#     )
    
#     # 7. Génération des rapports demandés
#     # 7.1 PDF avec tous les graphiques importants
#     create_report_graphics(train_df, test_df, test_results, optimized_model, X_train.columns)
    
#     # 7.2 PDF avec les statistiques de chaque modèle après entraînement
#     create_model_stats(train_results, "train")
    
#     # 7.3 PDF avec les statistiques de chaque modèle après test
#     create_model_stats(test_results, "test")
    
#     # 7.4 PDF avec le meilleur modèle et ses courbes
#     create_best_model_report(optimized_model, optimized_metrics, X_train.columns, best_model_name)
    
#     # 8. Sauvegarde du meilleur modèle
#     model_artifacts = {
#         'model': optimized_model,
#         'scaler': scaler,
#         'feature_names': list(X_train.columns),
#         'date_created': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
#     }
    
#     os.makedirs('models', exist_ok=True)
#     model_filename = f"models/heart_disease_{best_model_name.lower().replace(' ', '_')}_corrected.joblib"
#     joblib.dump(model_artifacts, model_filename)
#     print(f"\n✅ Modèle sauvegardé sous '{model_filename}'")
    
#     print("\n✅ Analyse terminée avec succès!")

# if __name__ == "__main__":
#     main()




"""
Script principal modifié avec optimisations pour améliorer la précision et l'exactitude
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
from informatiqueDecisionnelle.datamanagement import load_datasets
from informatiqueDecisionnelle.dataset import explore_datasets, visualize_data
from informatiqueDecisionnelle.features import prepare_data, balance_training_data, select_features
from informatiqueDecisionnelle.modeling.train import train_and_evaluate_models, compare_models, optimize_best_model, create_precision_focused_ensemble
from informatiqueDecisionnelle.modeling.predict import make_predictions
from informatiqueDecisionnelle.plots import create_report_graphics, create_model_stats, create_test_stats, create_best_model_report
"""
Script principal modifié avec optimisations pour améliorer la précision et l'exactitude
"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from informatiqueDecisionnelle.datamanagement import load_datasets
from informatiqueDecisionnelle.dataset import explore_datasets, visualize_data
from informatiqueDecisionnelle.features import prepare_data, balance_training_data, select_features


"""
Correctif pour ajouter les importations nécessaires à main.py
"""

# Ajoutez ces lignes au début de votre fichier main.py juste après les importations existantes
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc



def main():
    """
    Fonction principale qui exécute le workflow complet d'analyse et de prédiction
    avec des optimisations pour la précision et l'exactitude
    """
    print("\n" + "="*80)
    print(" ANALYSE PRÉDICTIVE DE MALADIES CARDIAQUES (PRÉCISION AMÉLIORÉE)")
    print("="*80)
    
    # 1. Chargement des datasets
    print("\n Chargement des datasets...")
    train_path = 'data/raw/heart_70pct.csv'
    test_path = 'data/raw/heart_30pct.csv'
    
    # Vérifier si les fichiers existent déjà
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print("Fichiers séparés non trouvés. Division du dataset original...")
        from split_data import split_and_save_data
        
        train_path, test_path = split_and_save_data(
            'data/raw/heart.csv', 
            train_ratio=0.615
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
    
    # 3. Préparation des données avec ingénierie des caractéristiques axée sur la précision
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    # 3.1 Ingénierie des caractéristiques de précision
    from informatiqueDecisionnelle.features import precision_focused_feature_engineering
    X_train_enhanced, X_test_enhanced = precision_focused_feature_engineering(X_train, X_test)
    
    # 3.2 Sélection des caractéristiques
    X_train_selected, X_test_selected = select_features(X_train_enhanced, X_test_enhanced, y_train, method='precision_focused')
    
    # 3.3 Équilibrage des classes optimisé pour la précision
    X_train_balanced, y_train_balanced, class_weights = balance_training_data(X_train_selected, y_train, method='precision_focused')
    
    # 3.4 Normalisation
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # 4. Entraînement et évaluation des modèles
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    
    # 4.1 Définition des modèles avec paramètres optimisés pour la précision
    models = {
        'Logistic Regression': LogisticRegression(
            C=0.8,                # Légère régularisation
            penalty='l2',
            solver='liblinear',
            class_weight=class_weights,
            max_iter=2000,
            random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=5,          # Limiter pour éviter le surapprentissage
            min_samples_split=5,  # Plus grand pour plus de stabilité
            min_samples_leaf=4,   # Plus grand pour meilleure généralisation
            class_weight=class_weights,
            random_state=42
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=9,       # Optimisé par résultats précédents
            weights='distance',  # Plus de poids aux voisins proches
            metric='minkowski',
            p=2,
            algorithm='auto'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            class_weight=class_weights,
            random_state=42
        ),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=200,
            learning_rate=0.8,
            algorithm='SAMME',  # Changé de 'SAMME.R' à 'SAMME'
            random_state=42     
        ),
        'SVM': SVC(
            C=10.0,              # Meilleure marge
            kernel='rbf',
            gamma='scale',
            class_weight=class_weights,
            probability=True,
            random_state=42
        )
    }
    
    # 4.2 Entraînement et évaluation des modèles
    test_results = {}
    train_results = {}
    
    for name, model in models.items():
        print(f"\n Entraînement et évaluation du modèle {name}...")
        
        # Entraînement
        model.fit(X_train_scaled, y_train_balanced)
        
        # Évaluation sur l'ensemble de test
        y_test_pred = model.predict(X_test_scaled)
        
        # Métriques de test
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        
        # Matrice de confusion
        test_cm = confusion_matrix(y_test, y_test_pred)
        if test_cm.shape == (2, 2):
            tn, fp, fn, tp = test_cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
        
        # Probabilités si disponibles
        if hasattr(model, 'predict_proba'):
            y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
            test_fpr, test_tpr, _ = roc_curve(y_test, y_test_proba)
            test_auc = auc(test_fpr, test_tpr)
            test_has_proba = True
        else:
            test_has_proba = False
            test_auc = 0
            test_fpr, test_tpr = None, None
            y_test_proba = None
        
        # Stockage des résultats de test
        test_results[name] = {
            'model': model,
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1,
            'confusion_matrix': test_cm,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            'has_proba': test_has_proba,
            'auc': test_auc,
            'fpr': test_fpr,
            'tpr': test_tpr,
            'y_test': y_test,
            'y_pred': y_test_pred,
            'y_proba': y_test_proba
        }
        
        # Évaluation sur l'ensemble d'entraînement
        y_train_pred = model.predict(X_train_scaled)
        
        # Métriques d'entraînement
        train_accuracy = accuracy_score(y_train_balanced, y_train_pred)
        train_precision = precision_score(y_train_balanced, y_train_pred)
        train_recall = recall_score(y_train_balanced, y_train_pred)
        train_f1 = f1_score(y_train_balanced, y_train_pred)
        
        # Matrice de confusion d'entraînement
        train_cm = confusion_matrix(y_train_balanced, y_train_pred)
        if train_cm.shape == (2, 2):
            train_tn, train_fp, train_fn, train_tp = train_cm.ravel()
        else:
            train_tn, train_fp, train_fn, train_tp = 0, 0, 0, 0
        
        # Probabilités pour l'entraînement
        if hasattr(model, 'predict_proba'):
            y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
            train_fpr, train_tpr, _ = roc_curve(y_train_balanced, y_train_proba)
            train_auc = auc(train_fpr, train_tpr)
            train_has_proba = True
        else:
            train_has_proba = False
            train_auc = 0
            train_fpr, train_tpr = None, None
            y_train_proba = None
        
        # Stockage des résultats d'entraînement
        train_results[name] = {
            'model': model,
            'accuracy': train_accuracy,
            'precision': train_precision,
            'recall': train_recall,
            'f1': train_f1,
            'confusion_matrix': train_cm,
            'tn': train_tn, 'fp': train_fp, 'fn': train_fn, 'tp': train_tp,
            'has_proba': train_has_proba,
            'auc': train_auc,
            'fpr': train_fpr,
            'tpr': train_tpr,
            'y_true': y_train_balanced,
            'y_pred': y_train_pred,
            'y_proba': y_train_proba
        }
        
        # Affichage des résultats
        print(f" Résultats pour {name}:")
        print(f"   - ENTRAINEMENT - Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")
        if train_has_proba:
            print(f"     AUC: {train_auc:.4f}")
        print(f"   - TEST - Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
        if test_has_proba:
            print(f"     AUC: {test_auc:.4f}")
        print(f"   - Matrice confusion (test): TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    # 5. Création d'un ensemble optimisé pour la précision
    precision_model, precision_metrics = create_precision_focused_ensemble(
        models, X_train_scaled, X_test_scaled, y_train_balanced, y_test, class_weights
    )
    
    # Ajouter ce modèle aux résultats
    test_results['Precision Ensemble'] = precision_metrics
    
    # 6. Création des rapports
    # 6.1 Rapport graphique
    create_report_graphics(train_df, test_df, test_results, precision_model, X_train_selected.columns)
    
    # 6.2 Statistiques des modèles
    create_model_stats(train_results, "train")
    create_model_stats(test_results, "test")
    
    # 6.3 Rapport du meilleur modèle
    create_best_model_report(precision_model, precision_metrics, X_train_selected.columns, "Precision Ensemble")
    
    # 7. Sauvegarde du meilleur modèle
    model_artifacts = {
        'model': precision_model,
        'base_models': precision_metrics['base_models'],
        'threshold': precision_metrics.get('threshold', 0.5),
        'scaler': scaler,
        'feature_names': list(X_train_selected.columns),
        'date_created': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model_artifacts, f"models/heart_disease_precision_ensemble.joblib")
    print(f"\n Modèle optimisé sauvegardé sous: models/heart_disease_precision_ensemble.joblib")
    
    # 8. Comparaison avec les résultats cibles
    target_metrics = {
        'accuracy': 0.92,
        'precision': 0.8696,
        'recall': 0.9062,
        'f1': 0.8889,
        'auc': 0.91
    }
    
    final_metrics = precision_metrics

    #metriques finales
    
    print("\n Comparaison avec les métriques cibles:")
    print(f"{'Métrique':<10} {'Cible':<10} {'Obtenue':<10} {'Différence':<10}")
    print("-"*40)
    for metric, target in target_metrics.items():
        obtained = final_metrics.get(metric, 0)
        diff = obtained - target
        result = "✅" if obtained >= target else "❌"
        print(f"{metric:<10} {target:.4f}    {obtained:.4f}    {diff:+.4f}    {result}")
    
    print("\n Analyse terminée avec succès!")

if __name__ == "__main__":
    main()