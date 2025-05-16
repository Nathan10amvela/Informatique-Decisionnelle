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
            print(f"⚠️ Matrice de confusion de forme inattendue pour l'entraînement: {train_conf_matrix.shape}")
        
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
        print(f"✅ Résultats pour {name}:")
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
    print("\n🔄 Création de modèles d'ensemble...")
    
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
        print("⏳ Entraînement du Voting Classifier (soft)...")
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
        print(f"✅ Résultats pour Voting Ensemble:")
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
            print("⏳ Entraînement du Stacking Classifier...")
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
            print(f"✅ Résultats pour Stacking Ensemble:")
            print(f"   - Accuracy (entrainement/test): {train_accuracy_stack:.4f}/{test_accuracy_stack:.4f}")
            print(f"   - Precision (entrainement/test): {train_precision_stack:.4f}/{test_precision_stack:.4f}")
            print(f"   - Recall (entrainement/test): {train_recall_stack:.4f}/{test_recall_stack:.4f}")
            print(f"   - F1-score (entrainement/test): {train_f1_stack:.4f}/{test_f1_stack:.4f}")
            print(f"   - AUC (entrainement/test): {train_roc_auc_stack:.4f}/{test_roc_auc_stack:.4f}")
            print(f"   - Matrice de confusion (test): TN={tn_test_stack}, FP={fp_test_stack}, FN={fn_test_stack}, TP={tp_test_stack}")
        
        except Exception as e:
            print(f"⚠️ Erreur lors de la création du Stacking Classifier: {e}")
    
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
    print("\n📊 Comparaison des performances des modèles:")
    
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
    
    print(f"\n🏆 Le meilleur modèle selon le F1-Score est {best_model_name} avec un score de {best_f1:.4f}")
    
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
            print("ℹ️ Optimisation d'un modèle d'ensemble...")
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
            
            print(f"✅ Modèle {best_model_name} entraîné avec succès!")
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
    
    print(f"⏳ Recherche des meilleurs hyperparamètres avec {cv_folds}-fold CV...")
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv_folds,
        scoring=scoring,
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"\n✅ Optimisation terminée!")
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
    print(f"\n📊 Performance du modèle optimisé sur l'ensemble de test:")
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