"""
Module pour l'entraînement et l'évaluation des modèles
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           roc_curve, auc, precision_recall_curve, confusion_matrix,
                           classification_report)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
#from xgboost import XGBClassifier
from sklearn.utils import class_weight
import os
import warnings
warnings.filterwarnings('ignore')

from ..config import MODELS_CONFIG, EVALUATION_CONFIG

def train_and_evaluate_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Entraîne et évalue plusieurs modèles de machine learning
    
    Args:
        X_train: Caractéristiques d'entraînement non normalisées
        X_test: Caractéristiques de test non normalisées
        X_train_scaled: Caractéristiques d'entraînement normalisées
        X_test_scaled: Caractéristiques de test normalisées
        y_train: Étiquettes d'entraînement
        y_test: Étiquettes de test
        
    Returns:
        Un tuple contenant (résultats sur les données de test, résultats sur les données d'entraînement)
    """
    print("\n🧠 Entraînement et évaluation des modèles...")
    
    # Création des modèles
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier(),
        'Random Forest': RandomForestClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    # Calcul des poids des classes pour gérer le déséquilibre
    use_class_weights = EVALUATION_CONFIG.get('use_class_weights', False)
    if use_class_weights:
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        
        # Application des poids aux modèles qui le supportent
        models['Logistic Regression'].set_params(class_weight=class_weight_dict)
        models['Decision Tree'].set_params(class_weight=class_weight_dict)
        models['SVM'].set_params(class_weight=class_weight_dict)
        models['Random Forest'].set_params(class_weight=class_weight_dict)
    
    # Dictionnaires pour stocker les résultats
    test_results = {}
    train_results = {}
    
    # Entraînement et évaluation de chaque modèle
    for name, model in models.items():
        print(f"\n⏳ Entraînement du modèle: {name}...")
        
        # Entraînement du modèle
        model.fit(X_train_scaled, y_train)
        
        # Validation croisée sur les données d'entraînement
        cv_folds = EVALUATION_CONFIG.get('cv_folds', 5)
        scoring = EVALUATION_CONFIG.get('scoring', 'f1')
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds, scoring=scoring)
        cv_score_mean = cv_scores.mean()
        cv_score_std = cv_scores.std()
        
        # ------- Évaluation sur les données d'entraînement -------
        y_train_pred = model.predict(X_train_scaled)
        
        # Matrice de confusion pour les données d'entraînement
        train_conf_matrix = confusion_matrix(y_train, y_train_pred)
        
        # Extraction des valeurs de la matrice de confusion
        if train_conf_matrix.shape == (2, 2):
            tn_train, fp_train, fn_train, tp_train = train_conf_matrix.ravel()
        else:
            tn_train, fp_train, fn_train, tp_train = 0, 0, 0, 0
            print(f" Matrice de confusion de forme inattendue pour l'entraînement: {train_conf_matrix.shape}")
        
        # Calcul des métriques d'entraînement
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred)
        train_recall = recall_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        train_report = classification_report(y_train, y_train_pred, output_dict=True)
        
        # Calcul des métriques par classe pour l'entraînement
        train_precision_per_class = {}
        train_recall_per_class = {}
        train_f1_per_class = {}
        
        for cls in np.unique(y_train):
            train_precision_per_class[cls] = train_report[str(cls)]['precision']
            train_recall_per_class[cls] = train_report[str(cls)]['recall']
            train_f1_per_class[cls] = train_report[str(cls)]['f1-score']
        
        # Calcul de l'AUC sur l'entraînement si disponible
        if hasattr(model, 'predict_proba'):
            y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
            train_fpr, train_tpr, _ = roc_curve(y_train, y_train_proba)
            train_roc_auc = auc(train_fpr, train_tpr)
            train_has_proba = True
        else:
            train_has_proba = False
            train_roc_auc = None
            train_fpr, train_tpr = None, None
            y_train_proba = None
            
        # ------- Évaluation sur les données de test -------
        y_test_pred = model.predict(X_test_scaled)
        
        # Matrice de confusion pour les données de test
        test_conf_matrix = confusion_matrix(y_test, y_test_pred)
        
        # Extraction des valeurs de la matrice de confusion
        if test_conf_matrix.shape == (2, 2):
            tn_test, fp_test, fn_test, tp_test = test_conf_matrix.ravel()
        else:
            tn_test, fp_test, fn_test, tp_test = 0, 0, 0, 0
            print(f"⚠️ Matrice de confusion de forme inattendue pour le test: {test_conf_matrix.shape}")
        
        # Calcul des métriques de test
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_report = classification_report(y_test, y_test_pred, output_dict=True)
        
        # Calcul des métriques par classe pour le test
        test_precision_per_class = {}
        test_recall_per_class = {}
        test_f1_per_class = {}
        
        for cls in np.unique(y_test):
            test_precision_per_class[cls] = test_report[str(cls)]['precision']
            test_recall_per_class[cls] = test_report[str(cls)]['recall']
            test_f1_per_class[cls] = test_report[str(cls)]['f1-score']
        
        # Calcul de l'AUC sur le test si disponible
        if hasattr(model, 'predict_proba'):
            y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
            test_fpr, test_tpr, _ = roc_curve(y_test, y_test_proba)
            test_roc_auc = auc(test_fpr, test_tpr)
            test_has_proba = True
        else:
            test_has_proba = False
            test_roc_auc = None
            test_fpr, test_tpr = None, None
            y_test_proba = None
        
        # Stockage des résultats d'entraînement
        train_results[name] = {
            'model': model,
            'cv_score_mean': cv_score_mean,
            'cv_score_std': cv_score_std,
            'accuracy': train_accuracy,
            'precision': train_precision,
            'recall': train_recall,
            'f1': train_f1,
            'precision_per_class': train_precision_per_class,
            'recall_per_class': train_recall_per_class,
            'f1_per_class': train_f1_per_class,
            'confusion_matrix': train_conf_matrix,
            'tn': tn_train, 'fp': fp_train, 'fn': fn_train, 'tp': tp_train,
            'report': train_report,
            'has_proba': train_has_proba,
            'auc': train_roc_auc if train_has_proba else 0,
            'fpr': train_fpr,
            'tpr': train_tpr,
            'y_true': y_train,
            'y_pred': y_train_pred,
            'y_proba': y_train_proba
        }
        
        # Stockage des résultats de test
        test_results[name] = {
            'model': model,
            'cv_score_mean': cv_score_mean,
            'cv_score_std': cv_score_std,
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1,
            'precision_per_class': test_precision_per_class,
            'recall_per_class': test_recall_per_class,
            'f1_per_class': test_f1_per_class,
            'confusion_matrix': test_conf_matrix,
            'tn': tn_test, 'fp': fp_test, 'fn': fn_test, 'tp': tp_test,
            'report': test_report,
            'has_proba': test_has_proba,
            'auc': test_roc_auc if test_has_proba else 0,
            'fpr': test_fpr,
            'tpr': test_tpr,
            'y_test': y_test,
            'y_pred': y_test_pred,
            'y_proba': y_test_proba
        }
        
        # Affichage des résultats de test (le plus important à surveiller)
        print(f" Résultats pour {name}:")
        print(f"   - Validation croisée ({cv_folds}-fold): {cv_score_mean:.4f} ± {cv_score_std:.4f}")
        print(f"   - Accuracy (entrainement/test): {train_accuracy:.4f}/{test_accuracy:.4f}")
        print(f"   - Precision (entrainement/test): {train_precision:.4f}/{test_precision:.4f}")
        print(f"   - Recall (entrainement/test): {train_recall:.4f}/{test_recall:.4f}")
        print(f"   - F1-score (entrainement/test): {train_f1:.4f}/{test_f1:.4f}")
        
        if test_has_proba:
            print(f"   - AUC (entrainement/test): {train_roc_auc:.4f}/{test_roc_auc:.4f}")
        
        print(f"   - Matrice de confusion (test): TN={tn_test}, FP={fp_test}, FN={fn_test}, TP={tp_test}")
    
    # Création d'un modèle d'ensemble (Voting Classifier)
    create_ensemble_model(models, test_results, train_results, X_train_scaled, X_test_scaled, y_train, y_test)
    
    return test_results, train_results

def create_ensemble_model(base_models, test_results, train_results, X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Crée un modèle d'ensemble à partir des modèles de base
    
    Args:
        base_models: Dictionnaire des modèles de base
        test_results: Dictionnaire pour stocker les résultats de test
        train_results: Dictionnaire pour stocker les résultats d'entraînement
        X_train_scaled, X_test_scaled: Données d'entraînement et de test
        y_train, y_test: Étiquettes d'entraînement et de test
    """
    print("\n Création de modèles d'ensemble...")
    
    # Filtrer les modèles qui supportent predict_proba pour le VotingClassifier
    prob_models = {}
    for name, model in base_models.items():
        if hasattr(model, 'predict_proba'):
            prob_models[name] = model
    
    if len(prob_models) >= 3:  # Au moins 3 modèles pour un ensemble significatif
        # Création d'un Voting Classifier avec les meilleurs modèles
        voting_soft = VotingClassifier(
            estimators=[(name, model) for name, model in prob_models.items()],
            voting='soft'
        )
        
        # Entraînement du modèle d'ensemble
        print(" Entraînement du Voting Classifier (soft)...")
        voting_soft.fit(X_train_scaled, y_train)
        
        # --- Évaluation sur les données d'entraînement ---
        y_train_pred = voting_soft.predict(X_train_scaled)
        y_train_proba = voting_soft.predict_proba(X_train_scaled)[:, 1]
        
        # Matrice de confusion pour l'entraînement
        train_conf_matrix = confusion_matrix(y_train, y_train_pred)
        
        # Extraction des valeurs de la matrice de confusion d'entraînement
        if train_conf_matrix.shape == (2, 2):
            tn_train, fp_train, fn_train, tp_train = train_conf_matrix.ravel()
        else:
            tn_train, fp_train, fn_train, tp_train = 0, 0, 0, 0
        
        # Calcul des métriques d'entraînement
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred)
        train_recall = recall_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        train_report = classification_report(y_train, y_train_pred, output_dict=True)
        
        # Calcul des métriques par classe pour l'entraînement
        train_precision_per_class = {}
        train_recall_per_class = {}
        train_f1_per_class = {}
        
        for cls in np.unique(y_train):
            train_precision_per_class[cls] = train_report[str(cls)]['precision']
            train_recall_per_class[cls] = train_report[str(cls)]['recall']
            train_f1_per_class[cls] = train_report[str(cls)]['f1-score']
        
        # Calcul de l'AUC pour l'entraînement
        train_fpr, train_tpr, _ = roc_curve(y_train, y_train_proba)
        train_roc_auc = auc(train_fpr, train_tpr)
        
        # --- Évaluation sur les données de test ---
        y_test_pred = voting_soft.predict(X_test_scaled)
        y_test_proba = voting_soft.predict_proba(X_test_scaled)[:, 1]
        
        # Matrice de confusion pour le test
        test_conf_matrix = confusion_matrix(y_test, y_test_pred)
        
        # Extraction des valeurs de la matrice de confusion de test
        if test_conf_matrix.shape == (2, 2):
            tn_test, fp_test, fn_test, tp_test = test_conf_matrix.ravel()
        else:
            tn_test, fp_test, fn_test, tp_test = 0, 0, 0, 0
        
        # Calcul des métriques de test
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_report = classification_report(y_test, y_test_pred, output_dict=True)
        
        # Calcul des métriques par classe pour le test
        test_precision_per_class = {}
        test_recall_per_class = {}
        test_f1_per_class = {}
        
        for cls in np.unique(y_test):
            test_precision_per_class[cls] = test_report[str(cls)]['precision']
            test_recall_per_class[cls] = test_report[str(cls)]['recall']
            test_f1_per_class[cls] = test_report[str(cls)]['f1-score']
        
        # Calcul de l'AUC pour le test
        test_fpr, test_tpr, _ = roc_curve(y_test, y_test_proba)
        test_roc_auc = auc(test_fpr, test_tpr)
        
        # Stockage des résultats d'entraînement
        train_results['Voting Ensemble'] = {
            'model': voting_soft,
            'cv_score_mean': np.mean([train_results[name]['cv_score_mean'] for name in prob_models]),
            'cv_score_std': np.mean([train_results[name]['cv_score_std'] for name in prob_models]),
            'accuracy': train_accuracy,
            'precision': train_precision,
            'recall': train_recall,
            'f1': train_f1,
            'precision_per_class': train_precision_per_class,
            'recall_per_class': train_recall_per_class,
            'f1_per_class': train_f1_per_class,
            'confusion_matrix': train_conf_matrix,
            'tn': tn_train, 'fp': fp_train, 'fn': fn_train, 'tp': tp_train,
            'report': train_report,
            'has_proba': True,
            'auc': train_roc_auc,
            'fpr': train_fpr,
            'tpr': train_tpr,
            'y_true': y_train,
            'y_pred': y_train_pred,
            'y_proba': y_train_proba
        }
        
        # Stockage des résultats de test
        test_results['Voting Ensemble'] = {
            'model': voting_soft,
            'cv_score_mean': np.mean([test_results[name]['cv_score_mean'] for name in prob_models]),
            'cv_score_std': np.mean([test_results[name]['cv_score_std'] for name in prob_models]),
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1,
            'precision_per_class': test_precision_per_class,
            'recall_per_class': test_recall_per_class,
            'f1_per_class': test_f1_per_class,
            'confusion_matrix': test_conf_matrix,
            'tn': tn_test, 'fp': fp_test, 'fn': fn_test, 'tp': tp_test,
            'report': test_report,
            'has_proba': True,
            'auc': test_roc_auc,
            'fpr': test_fpr,
            'tpr': test_tpr,
            'y_test': y_test,
            'y_pred': y_test_pred,
            'y_proba': y_test_proba
        }
        
        # Affichage des résultats
        print(f" Résultats pour Voting Ensemble:")
        print(f"   - Accuracy (entrainement/test): {train_accuracy:.4f}/{test_accuracy:.4f}")
        print(f"   - Precision (entrainement/test): {train_precision:.4f}/{test_precision:.4f}")
        print(f"   - Recall (entrainement/test): {train_recall:.4f}/{test_recall:.4f}")
        print(f"   - F1-score (entrainement/test): {train_f1:.4f}/{test_f1:.4f}")
        print(f"   - AUC (entrainement/test): {train_roc_auc:.4f}/{test_roc_auc:.4f}")
        print(f"   - Matrice de confusion (test): TN={tn_test}, FP={fp_test}, FN={fn_test}, TP={tp_test}")
        
        # Création d'un Stacking Classifier
        try:
            # Définition des modèles de base et du méta-classifieur
            estimators = [(name, model) for name, model in prob_models.items()]
            stacking = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(),
                cv=5,
                stack_method='predict_proba'
            )
            
            # Entraînement du modèle
            print(" Entraînement du Stacking Classifier...")
            stacking.fit(X_train_scaled, y_train)
            
            # --- Évaluation sur les données d'entraînement ---
            y_train_pred_stack = stacking.predict(X_train_scaled)
            y_train_proba_stack = stacking.predict_proba(X_train_scaled)[:, 1]
            
            # Matrice de confusion pour l'entraînement
            train_conf_matrix_stack = confusion_matrix(y_train, y_train_pred_stack)
            
            # Extraction des valeurs de la matrice de confusion d'entraînement
            if train_conf_matrix_stack.shape == (2, 2):
                tn_train_stack, fp_train_stack, fn_train_stack, tp_train_stack = train_conf_matrix_stack.ravel()
            else:
                tn_train_stack, fp_train_stack, fn_train_stack, tp_train_stack = 0, 0, 0, 0
            
            # Calcul des métriques d'entraînement
            train_accuracy_stack = accuracy_score(y_train, y_train_pred_stack)
            train_precision_stack = precision_score(y_train, y_train_pred_stack)
            train_recall_stack = recall_score(y_train, y_train_pred_stack)
            train_f1_stack = f1_score(y_train, y_train_pred_stack)
            train_report_stack = classification_report(y_train, y_train_pred_stack, output_dict=True)
            
            # Calcul des métriques par classe pour l'entraînement
            train_precision_per_class_stack = {}
            train_recall_per_class_stack = {}
            train_f1_per_class_stack = {}
            
            for cls in np.unique(y_train):
                train_precision_per_class_stack[cls] = train_report_stack[str(cls)]['precision']
                train_recall_per_class_stack[cls] = train_report_stack[str(cls)]['recall']
                train_f1_per_class_stack[cls] = train_report_stack[str(cls)]['f1-score']
            
            # Calcul de l'AUC pour l'entraînement
            train_fpr_stack, train_tpr_stack, _ = roc_curve(y_train, y_train_proba_stack)
            train_roc_auc_stack = auc(train_fpr_stack, train_tpr_stack)
            
            # --- Évaluation sur les données de test ---
            y_test_pred_stack = stacking.predict(X_test_scaled)
            y_test_proba_stack = stacking.predict_proba(X_test_scaled)[:, 1]
            
            # Matrice de confusion pour le test
            test_conf_matrix_stack = confusion_matrix(y_test, y_test_pred_stack)
            
            # Extraction des valeurs de la matrice de confusion de test
            if test_conf_matrix_stack.shape == (2, 2):
                tn_test_stack, fp_test_stack, fn_test_stack, tp_test_stack = test_conf_matrix_stack.ravel()
            else:
                tn_test_stack, fp_test_stack, fn_test_stack, tp_test_stack = 0, 0, 0, 0
            
            # Calcul des métriques de test
            test_accuracy_stack = accuracy_score(y_test, y_test_pred_stack)
            test_precision_stack = precision_score(y_test, y_test_pred_stack)
            test_recall_stack = recall_score(y_test, y_test_pred_stack)
            test_f1_stack = f1_score(y_test, y_test_pred_stack)
            test_report_stack = classification_report(y_test, y_test_pred_stack, output_dict=True)
            
            # Calcul des métriques par classe pour le test
            test_precision_per_class_stack = {}
            test_recall_per_class_stack = {}
            test_f1_per_class_stack = {}
            
            for cls in np.unique(y_test):
                test_precision_per_class_stack[cls] = test_report_stack[str(cls)]['precision']
                test_recall_per_class_stack[cls] = test_report_stack[str(cls)]['recall']
                test_f1_per_class_stack[cls] = test_report_stack[str(cls)]['f1-score']
            
            # Calcul de l'AUC pour le test
            test_fpr_stack, test_tpr_stack, _ = roc_curve(y_test, y_test_proba_stack)
            test_roc_auc_stack = auc(test_fpr_stack, test_tpr_stack)
            
            # Stockage des résultats d'entraînement
            train_results['Stacking Ensemble'] = {
                'model': stacking,
                'cv_score_mean': np.mean([train_results[name]['cv_score_mean'] for name in prob_models]),
                'cv_score_std': np.mean([train_results[name]['cv_score_std'] for name in prob_models]),
                'accuracy': train_accuracy_stack,
                'precision': train_precision_stack,
                'recall': train_recall_stack,
                'f1': train_f1_stack,
                'precision_per_class': train_precision_per_class_stack,
                'recall_per_class': train_recall_per_class_stack,
                'f1_per_class': train_f1_per_class_stack,
                'confusion_matrix': train_conf_matrix_stack,
                'tn': tn_train_stack, 'fp': fp_train_stack, 'fn': fn_train_stack, 'tp': tp_train_stack,
                'report': train_report_stack,
                'has_proba': True,
                'auc': train_roc_auc_stack,
                'fpr': train_fpr_stack,
                'tpr': train_tpr_stack,
                'y_true': y_train,
                'y_pred': y_train_pred_stack,
                'y_proba': y_train_proba_stack
            }
            
            # Stockage des résultats de test
            test_results['Stacking Ensemble'] = {
                'model': stacking,
                'cv_score_mean': np.mean([test_results[name]['cv_score_mean'] for name in prob_models]),
                'cv_score_std': np.mean([test_results[name]['cv_score_std'] for name in prob_models]),
                'accuracy': test_accuracy_stack,
                'precision': test_precision_stack,
                'recall': test_recall_stack,
                'f1': test_f1_stack,
                'precision_per_class': test_precision_per_class_stack,
                'recall_per_class': test_recall_per_class_stack,
                'f1_per_class': test_f1_per_class_stack,
                'confusion_matrix': test_conf_matrix_stack,
                'tn': tn_test_stack, 'fp': fp_test_stack, 'fn': fn_test_stack, 'tp': tp_test_stack,
                'report': test_report_stack,
                'has_proba': True,
                'auc': test_roc_auc_stack,
                'fpr': test_fpr_stack,
                'tpr': test_tpr_stack,
                'y_test': y_test,
                'y_pred': y_test_pred_stack,
                'y_proba': y_test_proba_stack
            }
            
            # Affichage des résultats
            print(f" Résultats pour Stacking Ensemble:")
            print(f"   - Accuracy (entrainement/test): {train_accuracy_stack:.4f}/{test_accuracy_stack:.4f}")
            print(f"   - Precision (entrainement/test): {train_precision_stack:.4f}/{test_precision_stack:.4f}")
            print(f"   - Recall (entrainement/test): {train_recall_stack:.4f}/{test_recall_stack:.4f}")
            print(f"   - F1-score (entrainement/test): {train_f1_stack:.4f}/{test_f1_stack:.4f}")
            print(f"   - AUC (entrainement/test): {train_roc_auc_stack:.4f}/{test_roc_auc_stack:.4f}")
            print(f"   - Matrice de confusion (test): TN={tn_test_stack}, FP={fp_test_stack}, FN={fn_test_stack}, TP={tp_test_stack}")
        
        except Exception as e:
            print(f" Erreur lors de la création du Stacking Classifier: {e}")
    
    else:
        print("⚠️ Pas assez de modèles supportant predict_proba pour créer un ensemble")
        
def compare_models(results, feature_names):
    """
    Compare les performances des différents modèles et identifie le meilleur
    
    Args:
        results: Dictionnaire avec les métriques des modèles
        feature_names: Noms des caractéristiques pour l'analyse d'importance
        
    Returns:
        Nom du meilleur modèle selon le F1-score
    """
    print("\n Comparaison des performances des modèles:")
    
    # Création d'un DataFrame pour la comparaison
    models_df = pd.DataFrame({
        'Modèle': list(results.keys()),
        'Accuracy': [results[model]['accuracy'] for model in results],
        'Precision': [results[model]['precision'] for model in results],
        'Recall': [results[model]['recall'] for model in results],
        'F1-Score': [results[model]['f1'] for model in results],
        'AUC': [results[model]['auc'] for model in results if results[model]['has_proba']]
    })
    
    print(models_df.sort_values('F1-Score', ascending=False))
    
    # Visualisation des métriques
    os.makedirs('reports/figures', exist_ok=True)
    
    # Vérifier qu'il y a des modèles à visualiser
    if len(models_df) == 0:
        print("⚠️ Aucun modèle à visualiser")
        return best_model_name
    
    plt.figure(figsize=(15, 12))
    
    # Limiter le nombre de modèles à afficher si nécessaire pour la lisibilité
    max_models_to_display = 8
    if len(models_df) > max_models_to_display:
        print(f"⚠️ Limitation à {max_models_to_display} modèles pour la visualisation")
        # Prioriser les modèles avec le meilleur F1-Score
        models_df_display = models_df.sort_values('F1-Score', ascending=False).head(max_models_to_display)
    else:
        models_df_display = models_df
    
    # 1. Comparaison des accuracies
    plt.subplot(2, 2, 1)
    sns.barplot(x='Modèle', y='Accuracy', data=models_df_display.sort_values('Accuracy', ascending=False))
    plt.title('Accuracy par modèle', fontsize=14)
    plt.ylim(0.5, 1.0)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 2. Comparaison des F1-Scores
    plt.subplot(2, 2, 2)
    sns.barplot(x='Modèle', y='F1-Score', data=models_df_display.sort_values('F1-Score', ascending=False))
    plt.title('F1-Score par modèle', fontsize=14)
    plt.ylim(0.5, 1.0)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 3. Précision vs Recall
    plt.subplot(2, 2, 3)
    models_df_melted = pd.melt(models_df_display, id_vars=['Modèle'], value_vars=['Precision', 'Recall'])
    sns.barplot(x='Modèle', y='value', hue='variable', data=models_df_melted)
    plt.title('Precision vs Recall par modèle', fontsize=14)
    plt.ylim(0.5, 1.0)
    plt.xticks(rotation=45)
    plt.legend(title='Métrique')
    plt.grid(True, alpha=0.3)
    
    # 4. Courbes ROC de tous les modèles
    plt.subplot(2, 2, 4)
    
    # Compter combien de modèles ont des données de probabilité
    prob_models = sum(1 for model_data in results.values() if model_data.get('has_proba', False))
    
    if prob_models > 0:
        for model_name, model_data in results.items():
            if model_data.get('has_proba', False) and 'fpr' in model_data and 'tpr' in model_data:
                plt.plot(model_data['fpr'], model_data['tpr'], lw=2, 
                        label=f"{model_name} (AUC = {model_data['auc']:.4f})")
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taux de faux positifs (1 - Spécificité)')
        plt.ylabel('Taux de vrais positifs (Sensibilité)')
        plt.title('Courbes ROC des différents modèles', fontsize=14)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, "Aucun modèle avec des probabilités disponible", 
                ha='center', va='center', fontsize=12)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('reports/figures/model_comparison.png')
    plt.close()
    
    # Identification du meilleur modèle selon le F1-score
    best_model_name = models_df.loc[models_df['F1-Score'].idxmax(), 'Modèle']
    best_f1 = models_df['F1-Score'].max()
    
    print(f"\n Le meilleur modèle selon le F1-Score est {best_model_name} avec un score de {best_f1:.4f}")
    
    # Analyse d'importance des caractéristiques pour les modèles compatibles
    for model_name, model_data in results.items():
        model = model_data['model']
        
        if hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_
            
            # Tri des caractéristiques par importance
            indices = np.argsort(feature_importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title(f'Importance des caractéristiques - {model_name}', fontsize=14)
            plt.barh(range(len(indices)), feature_importances[indices], align='center')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Importance relative')
            plt.tight_layout()
            plt.savefig(f'reports/figures/feature_importance_{model_name.replace(" ", "_").lower()}.png')
            plt.close()
    
    return best_model_name

def optimize_best_model(best_model_name, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Optimise les hyperparamètres du meilleur modèle
    
    Args:
        best_model_name: Nom du meilleur modèle
        X_train, X_test: Données non normalisées
        X_train_scaled, X_test_scaled: Données normalisées
        y_train, y_test: Étiquettes
        
    Returns:
        Tuple (modèle optimisé, métriques du modèle optimisé)
    """
    print(f"\n🔧 Optimisation des hyperparamètres pour {best_model_name}...")
    
    # Récupération de la configuration pour ce modèle
    if best_model_name in MODELS_CONFIG:
        param_grid = MODELS_CONFIG[best_model_name]['param_grid']
    else:
        # Si le modèle est un ensemble, utiliser une configuration par défaut
        if 'Ensemble' in best_model_name:
            print("ℹ Optimisation d'un modèle d'ensemble...")
            # Les ensembles nécessitent une approche spéciale pour l'optimisation
            # Pour simplifier, on retourne directement le modèle ensemble sans optimisation
            from sklearn.base import clone
            
            # Modèles de base
            models = {
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'KNN': KNeighborsClassifier(),
                'Random Forest': RandomForestClassifier(random_state=42),
                'AdaBoost': AdaBoostClassifier(random_state=42),
                'SVM': SVC(probability=True, random_state=42)
            }
            
            if best_model_name == 'Voting Ensemble':
                # Filtrer les modèles qui supportent predict_proba
                prob_models = {}
                for name, model in models.items():
                    if hasattr(model, 'predict_proba'):
                        prob_models[name] = model
                
                ensemble_model = VotingClassifier(
                    estimators=[(name, model) for name, model in prob_models.items()],
                    voting='soft'
                )
            elif best_model_name == 'Stacking Ensemble':
                prob_models = {}
                for name, model in models.items():
                    if hasattr(model, 'predict_proba'):
                        prob_models[name] = model
                
                ensemble_model = StackingClassifier(
                    estimators=[(name, model) for name, model in prob_models.items()],
                    final_estimator=LogisticRegression(),
                    cv=5,
                    stack_method='predict_proba'
                )
            else:
                raise ValueError(f"Type d'ensemble non reconnu: {best_model_name}")
            
            # Entraînement du modèle
            ensemble_model.fit(X_train_scaled, y_train)
            
            # Évaluation
            y_pred = ensemble_model.predict(X_test_scaled)
            y_proba = ensemble_model.predict_proba(X_test_scaled)[:, 1]
            
            # Métriques
            metrics = {
                'model': ensemble_model,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'has_proba': True,
                'auc': roc_auc_score(y_test, y_proba),
                'fpr': roc_curve(y_test, y_proba)[0],
                'tpr': roc_curve(y_test, y_proba)[1],
                'y_test': y_test,
                'y_pred': y_pred,
                'y_proba': y_proba
            }
            
            print(f" Modèle {best_model_name} entraîné avec succès!")
            print(f"   - Accuracy: {metrics['accuracy']:.4f}")
            print(f"   - Precision: {metrics['precision']:.4f}")
            print(f"   - Recall: {metrics['recall']:.4f}")
            print(f"   - F1-Score: {metrics['f1']:.4f}")
            print(f"   - AUC: {metrics['auc']:.4f}")
            
            return ensemble_model, metrics
        
        # Si le modèle n'est pas dans la configuration, utiliser une grille par défaut
        print(f"⚠️ Configuration non trouvée pour {best_model_name}, utilisation d'une grille par défaut")
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.1],
            'kernel': ['rbf', 'linear']
        }
    
    # Modèle à optimiser
    if best_model_name == 'Logistic Regression':
        base_model = LogisticRegression(random_state=42, max_iter=2000)
    elif best_model_name == 'Decision Tree':
        base_model = DecisionTreeClassifier(random_state=42)
    elif best_model_name == 'KNN':
        base_model = KNeighborsClassifier()
    elif best_model_name == 'Random Forest':
        base_model = RandomForestClassifier(random_state=42)
    elif best_model_name == 'AdaBoost':
        base_model = AdaBoostClassifier(random_state=42)
    elif best_model_name == 'SVM':
        base_model = SVC(probability=True, random_state=42)
    else:
        raise ValueError(f"Modèle non reconnu: {best_model_name}")
    
    # Optimisation par GridSearchCV
    cv_folds = EVALUATION_CONFIG.get('cv_folds', 5)
    scoring = EVALUATION_CONFIG.get('scoring', 'f1')
    
    print(f" Recherche des meilleurs hyperparamètres avec {cv_folds}-fold CV...")
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv_folds,
        scoring=scoring,
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"\n Optimisation terminée!")
    print(f"Meilleurs paramètres: {grid_search.best_params_}")
    print(f"Meilleur score CV ({scoring}): {grid_search.best_score_:.4f}")
    
    # Récupération du meilleur modèle
    best_model = grid_search.best_estimator_
    
    # Évaluation du modèle optimisé
    y_pred = best_model.predict(X_test_scaled)
    
    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Calcul des métriques par classe
    precision_per_class = {}
    recall_per_class = {}
    f1_per_class = {}
    
    for cls in np.unique(y_test):
        precision_per_class[cls] = report[str(cls)]['precision']
        recall_per_class[cls] = report[str(cls)]['recall']
        f1_per_class[cls] = report[str(cls)]['f1-score']
    
    # Calcul de l'AUC si disponible
    if hasattr(best_model, 'predict_proba'):
        y_proba = best_model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        has_proba = True
    else:
        has_proba = False
        roc_auc = None
        fpr, tpr = None, None
        y_proba = None
    
    # Stockage des métriques
    metrics = {
        'model': best_model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': conf_matrix,
        'report': report,
        'has_proba': has_proba,
        'auc': roc_auc if has_proba else 0,
        'fpr': fpr,
        'tpr': tpr,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba
    }
    
    # Affichage des résultats
    print(f"\n Performance du modèle optimisé sur l'ensemble de test:")
    print(f"   - Accuracy: {accuracy:.4f}")
    print(f"   - Precision: {precision:.4f}")
    print(f"   - Recall: {recall:.4f}")
    print(f"   - F1-score: {f1:.4f}")
    if has_proba:
        print(f"   - AUC: {roc_auc:.4f}")
    
    # Visualisation de la matrice de confusion
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Pas de maladie', 'Maladie'], 
               yticklabels=['Pas de maladie', 'Maladie'])
    plt.title(f'Matrice de confusion - {best_model_name} optimisé', fontsize=14)
    plt.xlabel('Prédit')
    plt.ylabel('Réel')
    plt.tight_layout()
    plt.savefig(f'reports/figures/confusion_matrix_optimized.png')
    plt.close()
    
    # Courbe ROC si disponible
    if has_proba:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='#e74c3c', lw=2, label=f'AUC = {roc_auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taux de faux positifs (1 - Spécificité)')
        plt.ylabel('Taux de vrais positifs (Sensibilité)')
        plt.title(f'Courbe ROC - {best_model_name} optimisé', fontsize=14)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'reports/figures/roc_curve_optimized.png')
        plt.close()
    
    return best_model, metrics



# Ajoutez cette fonction à la fin du fichier informatiqueDecisionnelle/modeling/train.py

def create_precision_focused_ensemble(base_models, X_train_scaled, X_test_scaled, y_train, y_test, class_weights=None):
    """
    Crée un ensemble optimisé pour la précision
    
    Args:
        base_models: Dictionnaire des modèles de base
        X_train_scaled, X_test_scaled: Données normalisées
        y_train, y_test: Étiquettes
        class_weights: Dictionnaire des poids des classes
        
    Returns:
        Modèle d'ensemble optimisé pour la précision, métriques
    """
    print("\n Création d'un ensemble optimisé pour la précision...")
    
    # Importer XGBoost si disponible
    try:
        from xgboost import XGBClassifier
        has_xgboost = True
    except ImportError:
        has_xgboost = False
        print("⚠️ XGBoost non disponible, utilisation des modèles de base uniquement")
    
    # Sélectionner uniquement les meilleurs modèles pour la précision
    selected_models = {}
    for name, model in base_models.items():
        # Prédictions sur l'ensemble d'entraînement
        y_pred = model.predict(X_train_scaled)
        precision = precision_score(y_train, y_pred)
        
        # Inclure seulement les modèles avec une bonne précision
        if precision > 0.75:
            selected_models[name] = model
            print(f"- Sélection du modèle {name} (précision: {precision:.4f})")
    
    if len(selected_models) < 2:
        print("⚠️ Trop peu de modèles avec bonne précision, inclusion de tous les modèles")
        selected_models = base_models.copy()
    
    # 1. Créer un meta-model optimisé pour la précision
    # Le meta-model doit être calibré pour maximiser la précision
    
    if has_xgboost:
        # XGBoost avec configuration axée sur la précision
        meta_model = XGBClassifier(
            learning_rate=0.05,
            n_estimators=300,
            max_depth=3,
            min_child_weight=3,  # Augmenter pour réduire les faux positifs
            gamma=0.2,           # Augmenter pour plus de régularisation
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            scale_pos_weight=0.8,  # < 1 pour favoriser la précision sur les classes positives
            random_state=42
        )
    else:
        # Random Forest avec paramètres axés sur la précision
        meta_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_split=5,  # Augmenter pour plus de stabilité
            min_samples_leaf=3,   # Augmenter pour réduire les faux positifs
            max_features='sqrt',
            bootstrap=True,
            random_state=42
        )
    
    # 2. Préparer les prédictions des modèles de base pour l'entraînement du meta-model
    from sklearn.model_selection import KFold
    
    # Utiliser validation croisée 5-fold pour obtenir des prédictions non biaisées
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Pour stocker les prédictions des modèles de base
    meta_features_train = np.zeros((X_train_scaled.shape[0], len(selected_models)))
    meta_features_test = np.zeros((X_test_scaled.shape[0], len(selected_models)))
    
    # Génération des meta-features
    for i, (name, model) in enumerate(selected_models.items()):
        # Pour les données de test, on peut prédire directement
        if hasattr(model, 'predict_proba'):
            meta_features_test[:, i] = model.predict_proba(X_test_scaled)[:, 1]
        else:
            meta_features_test[:, i] = model.predict(X_test_scaled)
        
        # Pour les données d'entraînement, utiliser la validation croisée
        meta_preds = np.zeros(X_train_scaled.shape[0])
        
        for train_idx, val_idx in kf.split(X_train_scaled):
            # Cloner le modèle pour l'entraîner sur chaque fold
            from sklearn.base import clone
            clone_model = clone(model)
            
            # Entraînement sur le sous-ensemble
            clone_model.fit(X_train_scaled[train_idx], y_train[train_idx])
            
            # Prédiction sur la validation
            if hasattr(clone_model, 'predict_proba'):
                meta_preds[val_idx] = clone_model.predict_proba(X_train_scaled[val_idx])[:, 1]
            else:
                meta_preds[val_idx] = clone_model.predict(X_train_scaled[val_idx])
        
        # Stocker les prédictions comme features
        meta_features_train[:, i] = meta_preds
    
    # 3. Entraîner le meta-model avec seuil de décision optimisé pour la précision
    
    print("- Entraînement du meta-modèle optimisé pour la précision...")
    meta_model.fit(meta_features_train, y_train)
    
    # Prédiction initiale
    if hasattr(meta_model, 'predict_proba'):
        y_proba = meta_model.predict_proba(meta_features_test)[:, 1]
        
        # Recherche du seuil optimal pour maximiser la précision tout en maintenant un bon rappel
        thresholds = np.linspace(0.3, 0.7, 20)
        best_f2 = 0  # F2-score donne plus de poids au rappel qu'à la précision
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred_t = (y_proba >= threshold).astype(int)
            prec = precision_score(y_test, y_pred_t)
            rec = recall_score(y_test, y_pred_t)
            
            # F2-score: équilibre entre précision et rappel avec plus de poids au rappel
            f2 = (5 * prec * rec) / (4 * prec + rec) if (prec + rec) > 0 else 0
            
            if f2 > best_f2:
                best_f2 = f2
                best_threshold = threshold
        
        print(f"- Seuil optimal pour la décision: {best_threshold:.3f}")
        y_pred = (y_proba >= best_threshold).astype(int)
    else:
        y_pred = meta_model.predict(meta_features_test)
        y_proba = None
    
    # 4. Évaluation du modèle
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Calcul de l'AUC si disponible
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        has_proba = True
    else:
        fpr, tpr, roc_auc = None, None, None
        has_proba = False
    
    # Matrice de confusion
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Extraire TP, TN, FP, FN
    if conf_matrix.shape == (2, 2):
        tn, fp, fn, tp = conf_matrix.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
    
    # Résultats
    print(f" Ensemble optimisé pour la précision:")
    print(f"   - Accuracy: {accuracy:.4f}")
    print(f"   - Precision: {precision:.4f}")
    print(f"   - Recall: {recall:.4f}")
    print(f"   - F1-score: {f1:.4f}")
    if has_proba:
        print(f"   - AUC: {roc_auc:.4f}")
    print(f"   - Matrice de confusion: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    # Visualisation de la courbe de précision en fonction du seuil
    if y_proba is not None:
        plt.figure(figsize=(10, 6))
        precisions = []
        recalls = []
        thresholds_viz = np.linspace(0.1, 0.9, 50)
        
        for threshold in thresholds_viz:
            y_pred_t = (y_proba >= threshold).astype(int)
            prec = precision_score(y_test, y_pred_t)
            rec = recall_score(y_test, y_pred_t)
            precisions.append(prec)
            recalls.append(rec)
        
        plt.plot(thresholds_viz, precisions, 'b-', label='Precision')
        plt.plot(thresholds_viz, recalls, 'r-', label='Recall')
        plt.axvline(x=best_threshold, color='g', linestyle='--', label=f'Seuil optimal: {best_threshold:.3f}')
        plt.xlabel('Seuil de décision')
        plt.ylabel('Score')
        plt.title('Précision et Rappel en fonction du seuil de décision')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('reports/figures/precision_threshold_optimization.png')
        plt.close()
    
    # Stockage des métriques
    metrics = {
        'model': meta_model,
        'base_models': selected_models,
        'meta_features_train': meta_features_train,
        'meta_features_test': meta_features_test,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'has_proba': has_proba,
        'auc': roc_auc if has_proba else 0,
        'fpr': fpr,
        'tpr': tpr,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'threshold': best_threshold if y_proba is not None else None
    }
    
    return meta_model, metrics


def train_regularized_models(X_train, X_test, y_train, y_test, feature_names):
    """
    Entraîne des modèles avec une forte régularisation pour éviter le surapprentissage
    
    Args:
        X_train, X_test: Données préparées
        y_train, y_test: Étiquettes
        feature_names: Noms des caractéristiques
    
    Returns:
        test_results, train_results: Résultats d'évaluation
    """
    print("\n Entraînement des modèles avec régularisation...")
    
    # Définition des modèles avec forte régularisation
    models = {
        'Logistic Regression': LogisticRegression(
            C=0.1,                 # Forte régularisation L2
            penalty='l2',
            solver='liblinear',
            max_iter=2000,
            random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=3,           # Profondeur très limitée
            min_samples_split=10,  # Valeur élevée pour éviter les splits trop spécifiques
            min_samples_leaf=5,    # Minimum de points dans chaque feuille
            random_state=42
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=11,        # Valeur plus élevée, moins sensible aux outliers
            weights='uniform',     # Poids uniforme plutôt que distance
            metric='euclidean'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,      # Nombre modéré d'arbres
            max_depth=4,           # Profondeur très limitée
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',   # Limiter les caractéristiques par split
            bootstrap=True,
            random_state=42
        ),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=50,       # Moins d'estimateurs
            learning_rate=0.01,    # Taux d'apprentissage très bas
            random_state=42
        ),
        'SVM': SVC(
            C=0.5,                 # Forte régularisation
            kernel='rbf',
            gamma='scale',
            probability=True,
            random_state=42
        )
    }
    
    # Évaluation avec validation croisée
    test_results = {}
    train_results = {}
    
    for name, model in models.items():
        print(f"\n Entraînement et évaluation de {name}...")
        
        # Validation croisée pour une évaluation plus fiable
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Entraînement sur tout l'ensemble d'entraînement
        model.fit(X_train, y_train)
        
        # Évaluation sur entraînement
        y_train_pred = model.predict(X_train)
        
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred)
        train_recall = recall_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        
        train_cm = confusion_matrix(y_train, y_train_pred)
        if len(train_cm) == 2:
            tn, fp, fn, tp = train_cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
        
        # Probabilités pour ROC (si disponible)
        train_has_proba = hasattr(model, "predict_proba")
        if train_has_proba:
            y_train_proba = model.predict_proba(X_train)[:, 1]
            train_fpr, train_tpr, _ = roc_curve(y_train, y_train_proba)
            train_auc = auc(train_fpr, train_tpr)
        else:
            train_fpr, train_tpr, train_auc = None, None, None
            y_train_proba = None
        
        # Évaluation sur test
        y_test_pred = model.predict(X_test)
        
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        
        test_cm = confusion_matrix(y_test, y_test_pred)
        if len(test_cm) == 2:
            test_tn, test_fp, test_fn, test_tp = test_cm.ravel()
        else:
            test_tn, test_fp, test_fn, test_tp = 0, 0, 0, 0
        
        # Probabilités pour ROC (si disponible)
        test_has_proba = hasattr(model, "predict_proba")
        if test_has_proba:
            y_test_proba = model.predict_proba(X_test)[:, 1]
            test_fpr, test_tpr, _ = roc_curve(y_test, y_test_proba)
            test_auc = auc(test_fpr, test_tpr)
        else:
            test_fpr, test_tpr, test_auc = None, None, None
            y_test_proba = None
        
        # Stockage des résultats d'entraînement
        train_results[name] = {
            'model': model,
            'cv_score_mean': cv_mean,
            'cv_score_std': cv_std,
            'accuracy': train_accuracy,
            'precision': train_precision,
            'recall': train_recall,
            'f1': train_f1,
            'confusion_matrix': train_cm,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            'has_proba': train_has_proba,
            'auc': train_auc if train_has_proba else 0,
            'fpr': train_fpr,
            'tpr': train_tpr,
            'y_true': y_train,
            'y_pred': y_train_pred,
            'y_proba': y_train_proba
        }
        
        # Stockage des résultats de test
        test_results[name] = {
            'model': model,
            'cv_score_mean': cv_mean,
            'cv_score_std': cv_std,
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1,
            'confusion_matrix': test_cm,
            'tn': test_tn, 'fp': test_fp, 'fn': test_fn, 'tp': test_tp,
            'has_proba': test_has_proba,
            'auc': test_auc if test_has_proba else 0,
            'fpr': test_fpr,
            'tpr': test_tpr,
            'y_test': y_test,
            'y_pred': y_test_pred,
            'y_proba': y_test_proba
        }
        
        # Affichage des résultats
        print(f" Résultats pour {name}:")
        print(f"   - CV (5-fold): {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"   - Train : Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")
        print(f"   - Test  : Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
        print(f"   - Écart  : Accuracy: {train_accuracy-test_accuracy:.4f}, F1: {train_f1-test_f1:.4f}")
        
        if train_has_proba:
            print(f"   - AUC   : Train: {train_auc:.4f}, Test: {test_auc:.4f}")
    
    # Ajouter un modèle d'ensemble simple basé sur le vote
    from sklearn.ensemble import VotingClassifier
    
    # Sélectionner uniquement les modèles avec predict_proba
    prob_models = {}
    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            prob_models[name] = model
    
    if len(prob_models) >= 3:
        print("\n Création d'un modèle d'ensemble simple par vote...")
        
        ensemble = VotingClassifier(
            estimators=[(name, model) for name, model in prob_models.items()],
            voting='soft'
        )
        
        ensemble.fit(X_train, y_train)
        
        # Évaluation sur entraînement
        y_train_pred = ensemble.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred)
        train_recall = recall_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        
        # Évaluation sur test
        y_test_pred = ensemble.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        
        # Probabilités
        y_train_proba = ensemble.predict_proba(X_train)[:, 1]
        train_fpr, train_tpr, _ = roc_curve(y_train, y_train_proba)
        train_auc = auc(train_fpr, train_tpr)
        
        y_test_proba = ensemble.predict_proba(X_test)[:, 1]
        test_fpr, test_tpr, _ = roc_curve(y_test, y_test_proba)
        test_auc = auc(test_fpr, test_tpr)
        
        # Matrices de confusion
        train_cm = confusion_matrix(y_train, y_train_pred)
        if len(train_cm) == 2:
            tn, fp, fn, tp = train_cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
            
        test_cm = confusion_matrix(y_test, y_test_pred)
        if len(test_cm) == 2:
            test_tn, test_fp, test_fn, test_tp = test_cm.ravel()
        else:
            test_tn, test_fp, test_fn, test_tp = 0, 0, 0, 0
        
        # Stocker les résultats
        train_results['Voting Ensemble'] = {
            'model': ensemble,
            'cv_score_mean': np.mean([train_results[m]['cv_score_mean'] for m in prob_models]),
            'cv_score_std': np.mean([train_results[m]['cv_score_std'] for m in prob_models]),
            'accuracy': train_accuracy,
            'precision': train_precision,
            'recall': train_recall,
            'f1': train_f1,
            'confusion_matrix': train_cm,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            'has_proba': True,
            'auc': train_auc,
            'fpr': train_fpr,
            'tpr': train_tpr,
            'y_true': y_train,
            'y_pred': y_train_pred,
            'y_proba': y_train_proba
        }
        
        test_results['Voting Ensemble'] = {
            'model': ensemble,
            'cv_score_mean': np.mean([test_results[m]['cv_score_mean'] for m in prob_models]),
            'cv_score_std': np.mean([test_results[m]['cv_score_std'] for m in prob_models]),
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1,
            'confusion_matrix': test_cm,
            'tn': test_tn, 'fp': test_fp, 'fn': test_fn, 'tp': test_tp,
            'has_proba': True,
            'auc': test_auc,
            'fpr': test_fpr,
            'tpr': test_tpr,
            'y_test': y_test,
            'y_pred': y_test_pred,
            'y_proba': y_test_proba
        }
        
        print(f" Résultats pour Voting Ensemble:")
        print(f"   - Train : Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")
        print(f"   - Test  : Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
        print(f"   - Écart  : Accuracy: {train_accuracy-test_accuracy:.4f}, F1: {train_f1-test_f1:.4f}")
        print(f"   - AUC   : Train: {train_auc:.4f}, Test: {test_auc:.4f}")
    
    return test_results, train_results

def select_best_model_by_generalization(test_results, train_results, feature_names):
    """
    Sélectionne le meilleur modèle basé sur la généralisation plutôt que sur la performance pure
    
    Args:
        test_results, train_results: Dictionnaires de résultats
        feature_names: Noms des caractéristiques
        
    Returns:
        Nom du meilleur modèle
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    
    print("\n Évaluation des modèles basée sur la généralisation:")
    
    # Création des données pour comparer
    comparison_data = []
    
    for name in test_results.keys():
        train_metrics = train_results[name]
        test_metrics = test_results[name]
        
        # Calcul des écarts entre train et test (mesure de surapprentissage)
        accuracy_gap = train_metrics['accuracy'] - test_metrics['accuracy']
        f1_gap = train_metrics['f1'] - test_metrics['f1']
        
        # Score composite valorisant à la fois performance et généralisation
        # Formula: test_f1 - 0.7 * f1_gap 
        # Cette formule pénalise fortement le surapprentissage tout en récompensant la performance sur test
        generalization_score = test_metrics['f1'] - 0.7 * f1_gap
        
        comparison_data.append({
            'Modèle': name,
            'Accuracy (train)': train_metrics['accuracy'],
            'Accuracy (test)': test_metrics['accuracy'],
            'F1 (train)': train_metrics['f1'],
            'F1 (test)': test_metrics['f1'],
            'Écart Accuracy': accuracy_gap,
            'Écart F1': f1_gap,
            'Score Généralisation': generalization_score
        })
    
    # Création du DataFrame et tri
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Score Généralisation', ascending=False)
    
    print(comparison_df)
    
    # Visualisation des écarts entre train et test
    plt.figure(figsize=(12, 6))
    
    # Limiter aux 5 premiers modèles pour la lisibilité
    plot_df = comparison_df.head(5)
    
    # Graphique barres empilées pour F1
    plt.subplot(1, 2, 1)
    plt.bar(plot_df['Modèle'], plot_df['F1 (test)'], label='F1 (test)')
    plt.bar(plot_df['Modèle'], plot_df['Écart F1'], bottom=plot_df['F1 (test)'], 
            label='Écart F1', color='lightgray', alpha=0.7)
    plt.title('F1-Score et écart Train/Test')
    plt.legend()
    plt.xticks(rotation=45)
    
    # Graphique des scores de généralisation
    plt.subplot(1, 2, 2)
    plt.bar(plot_df['Modèle'], plot_df['Score Généralisation'], color='green')
    plt.title('Score de Généralisation')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    os.makedirs('reports/figures', exist_ok=True)
    plt.savefig('reports/figures/model_generalization_comparison.png')
    plt.close()
    
    # Sélection du meilleur modèle selon le score de généralisation
    best_model_name = comparison_df.iloc[0]['Modèle']
    best_gen_score = comparison_df.iloc[0]['Score Généralisation']
    
    print(f"\n Meilleur modèle selon la généralisation: {best_model_name}")
    print(f"   Score de généralisation: {best_gen_score:.4f}")
    print(f"   F1 test: {comparison_df.iloc[0]['F1 (test)']:.4f}")
    print(f"   Écart F1: {comparison_df.iloc[0]['Écart F1']:.4f}")
    
    # Afficher l'importance des caractéristiques si disponible
    best_model = test_results[best_model_name]['model']
    
    if hasattr(best_model, 'feature_importances_'):
        print("\n Importance des caractéristiques du meilleur modèle:")
        
        # Si c'est un VotingClassifier
        if hasattr(best_model, 'estimators_') and hasattr(best_model, 'named_estimators_'):
            # VotingClassifier a des named_estimators_
            for name, estimator in best_model.named_estimators_.items():
                if hasattr(estimator, 'feature_importances_'):
                    print(f"\n   Importance des caractéristiques pour {name}:")
                    importances = estimator.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    
                    for i, idx in enumerate(indices):
                        if i < len(feature_names):  # Vérification pour éviter les erreurs d'index
                            print(f"   {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
                    break
        
        # Si c'est un RandomForestClassifier ou autre modèle avec feature_importances_
        elif hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            for i, idx in enumerate(indices):
                if i < len(feature_names):  # Vérification pour éviter les erreurs d'index
                    print(f"   {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    return best_model_name