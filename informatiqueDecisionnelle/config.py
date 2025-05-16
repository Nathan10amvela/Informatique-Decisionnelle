"""
Module de configuration pour le projet de prédiction des maladies cardiaques
"""

# Chemins des données
DATA_RAW_PATH = "data/raw"
DATA_PROCESSED_PATH = "data/processed"
MODELS_PATH = "models"
REPORTS_PATH = "reports"
FIGURES_PATH = "reports/figures"

# Configuration des modèles
MODELS_CONFIG = {
    'Logistic Regression': {
        'model_name': 'LR',
        'param_grid': {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['liblinear', 'lbfgs', 'saga'],
            'max_iter': [1000, 2000]
        }
    },
    'Decision Tree': {
        'model_name': 'DT',
        'param_grid': {
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4, 8],
            'criterion': ['gini', 'entropy']
        }
    },
    'KNN': {
        'model_name': 'KNN',
        'param_grid': {
            'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 100],  # 3, 5, 7, 9,
            'weights': [ 'distance' , 'uniform'],
            'metric': [ 'euclidean', 'manhattan', 'minkowski'],  # 
            'p': [ 1, 2, 3 , 4 , 5 , 25 , 30]  # Paramètre pour Minkowski  1, 2, 3 , 4 , 5 , 
        }
    },
    'Random Forest': {
        'model_name': 'RF',
        'param_grid': {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False],
            'class_weight': [None, 'balanced', 'balanced_subsample']
        }
    },
    'AdaBoost': {
        'model_name': 'AdaBoost',
        'param_grid': {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.5, 1.0, 1.5],
            'algorithm': ['SAMME', 'SAMME.R']
        }
    },
    'SVM': {
        'model_name': 'SVM',
        'param_grid': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
            'kernel': ['rbf', 'linear', 'poly'],
            'class_weight': [None, 'balanced'],
            'probability': [True]
        }
    }
}

# Configuration de l'exploration des données
EXPLORATION_CONFIG = {
    'correlation_threshold': 0.7,
    'missing_value_threshold': 0.05,
    'outlier_detection_method': 'iqr',
    'outlier_threshold': 1.5
}

# Configuration de la préparation des données
DATA_PREP_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'feature_engineering': True,
    'balance_classes': True,
    'feature_selection': True,
    'feature_selection_method': 'combined',
    'n_features_to_select': 10,
    'missing_value_strategy': 'median'
}

# Configuration de l'évaluation
EVALUATION_CONFIG = {
    'cv_folds': 5,
    'scoring': 'f1',
    'use_class_weights': True
}

# Couleurs personnalisées pour les visualisations
COLORS = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

# Configuration des rapports
REPORT_CONFIG = {
    'dpi': 300,
    'figure_extension': 'png',
    'pdf_title': 'Analyse et Prédiction des Maladies Cardiaques',
    'date': 'Avril 2025'
}

# Référence aux résultats précédents (à adapter selon les résultats précédents)
PREVIOUS_RESULTS = {
    'LR': {'accuracy': 0.9028, 'precision': 0.8667, 'recall': 0.8938, 'f1': 0.8800, 'auc': 0.8900},
    'DT': {'accuracy': 0.8194, 'precision': 0.8125, 'recall': 0.8469, 'f1': 0.8300, 'auc': 0.8500},
    'KNN': {'accuracy': 0.8333, 'precision': 0.8667, 'recall': 0.8938, 'f1': 0.8800, 'auc': 0.8900},
    'RF': {'accuracy': 0.9028, 'precision': 0.8511, 'recall': 0.8906, 'f1': 0.8700, 'auc': 0.8900},
    'AdaBoost': {'accuracy': 0.9028, 'precision': 0.8667, 'recall': 0.8938, 'f1': 0.8800, 'auc': 0.8900},
    'SVM': {'accuracy': 0.9200, 'precision': 0.8696, 'recall': 0.9062, 'f1': 0.8889, 'auc': 0.9100}
}