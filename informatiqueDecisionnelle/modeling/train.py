import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, classification_report, confusion_matrix, 
                           roc_curve, auc, roc_auc_score)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from informatiqueDecisionnelle.config import MODEL_PARAMS
from informatiqueDecisionnelle.plots import plot_model_performance
import joblib
from pathlib import Path


# Modèles disponibles
MODELS = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Entraîne et évalue tous les modèles"""
    results = {}
    
    for name, model in MODELS.items():
        model, metrics = evaluate_model(
            model, X_train, X_test, y_train, y_test, name
        )
        results[name] = metrics
    
    # Comparaison des modèles
    performance_df = plot_model_performance(results)
    best_model_name = performance_df.iloc[0]['Modèle']
    
    return results, best_model_name

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Évalue un modèle individuel"""
    print(f"\n🧠 Entraînement et évaluation du modèle {model_name}...")
    
    # Entraînement du modèle
    model.fit(X_train, y_train)
    
    # Validation croisée
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_accuracy = cv_scores.mean()
    
    # Prédictions
    y_pred = model.predict(X_test)
    
    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Stockage des résultats
    metrics = {
        'model': model,
        'accuracy_cv': cv_accuracy,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) if hasattr(model, 'predict_proba') else 0,
        'has_proba': hasattr(model, 'predict_proba')
    }
    
    return model, metrics

def optimize_model(model_name, X_train, X_test, y_train, y_test):
    """Optimise le meilleur modèle"""
    print(f"\n🔧 Optimisation des hyperparamètres pour {model_name}...")
    
    grid_search = GridSearchCV(
        MODELS[model_name],
        MODEL_PARAMS[model_name],
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    optimized_model = grid_search.best_estimator_
    
    # Évaluation du modèle optimisé
    y_pred = optimized_model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    print(f"\n✅ Optimisation terminée!")
    print(f"Meilleurs paramètres: {grid_search.best_params_}")
    print(f"Meilleur F1-score en validation croisée: {grid_search.best_score_:.4f}")
    print(f"\n📊 Performance du modèle optimisé sur l'ensemble de test:")
    for metric, value in metrics.items():
        print(f"- {metric.capitalize()}: {value:.4f}")
    
    return optimized_model

def save_model(model, scaler, feature_names, model_name):
    """Sauvegarde le modèle entraîné"""
    models_dir = Path(__file__).parent.parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    model_artifacts = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'date_created': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    joblib.dump(model_artifacts, models_dir / f"{model_name}.joblib")
    print(f"✅ Modèle sauvegardé sous 'models/{model_name}.joblib'")

