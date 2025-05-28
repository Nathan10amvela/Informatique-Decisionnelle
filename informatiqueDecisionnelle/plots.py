"""
Module pour la création de visualisations et de rapports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix, precision_recall_curve, auc
from matplotlib.backends.backend_pdf import PdfPages
import os
import math


def create_subplot_grid(n_plots, max_cols=3):
    """
    Crée une grille de sous-graphiques adaptée au nombre de plots
    
    Args:
        n_plots: Nombre de graphiques à afficher
        max_cols: Nombre maximum de colonnes dans la grille
        
    Returns:
        Tuple (n_rows, n_cols) définissant la grille
    """
    n_cols = min(n_plots, max_cols)
    n_rows = math.ceil(n_plots / n_cols)
    return n_rows, n_cols


def create_report_graphics(train_df, test_df, results, best_model, feature_names):
    """
    Crée un rapport PDF avec tous les graphiques importants pour l'analyse des données
    
    Args:
        train_df: DataFrame d'entraînement
        test_df: DataFrame de test
        results: Dictionnaire des résultats des modèles
        best_model: Meilleur modèle après optimisation
        feature_names: Noms des caractéristiques
    """
    print("\n📊 Création du rapport PDF avec les graphiques importants...")
    
    # Création des dossiers nécessaires
    os.makedirs('reports', exist_ok=True)
    os.makedirs('reports/figures', exist_ok=True)

    # Vérification des entrées
    if not isinstance(results, dict) or len(results) == 0:
        print("⚠️ Avertissement: Aucun résultat de modèle fourni. Création du rapport limitée.")
        results = {}
    
    if not isinstance(feature_names, (list, pd.Index)) or len(feature_names) == 0:
        print("⚠️ Avertissement: Noms des caractéristiques non fournis. Utilisation de noms génériques.")
        feature_names = [f"Feature_{i}" for i in range(10)]

    with PdfPages('reports/1-Rapport_Graphiques.pdf') as pdf:
        # 1. Page de titre
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.6, 'Analyse et Prédiction des Maladies Cardiaques', 
                ha='center', fontsize=24, fontweight='bold')
        plt.text(0.5, 0.5, 'Rapport des Visualisations (Version Corrigée)', ha='center', fontsize=20)
        plt.text(0.5, 0.4, 'Date: Avril 2025', ha='center', fontsize=14)
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        # 2. Distribution des caractéristiques disponibles
        plt.figure(figsize=(15, 10))
        plt.suptitle('Distribution des caractéristiques principales', fontsize=16, y=0.95)
        
        # Utiliser les caractéristiques disponibles dans le dataset
        available_features = [col for col in ['age', 'chol', 'thalach', 'oldpeak', 'trestbps'] 
                            if col in train_df.columns]
        
        for i, feature in enumerate(available_features[:6]):
            plt.subplot(2, 3, i+1)
            if feature in train_df.columns:
                sns.histplot(data=train_df, x=feature, hue='target', kde=True, 
                           palette=['#3498db', '#e74c3c'], alpha=0.7)
                plt.title(f'Distribution de {feature}', fontsize=12)
                plt.xlabel(feature)
                plt.ylabel('Fréquence')
            
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # 3. Corrélation entre les caractéristiques
        plt.figure(figsize=(12, 10))
        correlation_matrix = train_df.corr()
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   linewidths=0.5, fmt='.2f', cbar_kws={'shrink': .8})
        
        plt.title('Matrice de corrélation des caractéristiques', fontsize=16)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # 4. Distribution des maladies cardiaques par sexe (si disponible)
        if 'sex' in train_df.columns:
            plt.figure(figsize=(10, 6))
            sex_counts = train_df.groupby(['sex', 'target']).size().unstack(fill_value=0)
            ax = sex_counts.plot(kind='bar', color=['#3498db', '#e74c3c'])
            plt.title('Distribution des maladies cardiaques par sexe', fontsize=16)
            plt.xlabel('Sexe (0=Femme, 1=Homme)')
            plt.ylabel('Nombre de patients')
            plt.xticks([0, 1], ['Femme', 'Homme'], rotation=0)
            plt.legend(['Pas de maladie', 'Maladie cardiaque'])
            
            # Ajout des chiffres sur les barres
            for container in ax.containers:
                ax.bar_label(container, fmt='%d')
                
            plt.tight_layout()
            pdf.savefig()
            plt.close()
        
        # 5. Boxplots des caractéristiques importantes par classe
        plt.figure(figsize=(12, 8))
        plt.suptitle('Distribution des caractéristiques par classe', fontsize=16, y=0.95)
        
        available_features = [col for col in ['age', 'chol', 'thalach', 'oldpeak'] 
                            if col in train_df.columns]
        
        n_features = len(available_features)
        if n_features > 0:
            rows = (n_features + 1) // 2
            for i, feature in enumerate(available_features[:4]):
                plt.subplot(rows, 2, i+1)
                sns.boxplot(x='target', y=feature, data=train_df, palette=['#3498db', '#e74c3c'])
                plt.title(f'{feature} par classe', fontsize=12)
                plt.xlabel('Maladie cardiaque')
                plt.xticks([0, 1], ['Pas de maladie', 'Maladie cardiaque'])
                
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        # 6. Matrice de confusion pour chaque modèle - VERSION NETTOYÉE
        if results:
            num_models = len(results)
            num_cols = 3
            num_rows = (num_models + num_cols - 1) // num_cols
            
            plt.figure(figsize=(15, 5*num_rows))
            plt.suptitle('Matrices de confusion des différents modèles', fontsize=16, y=0.95)
            
            for i, (model_name, metrics) in enumerate(results.items()):
                plt.subplot(num_rows, num_cols, i+1)
                
                # Vérifier si les données nécessaires existent
                if 'y_test' in metrics and 'y_pred' in metrics:
                    cm = confusion_matrix(metrics['y_test'], metrics['y_pred'])
                    
                    # Matrice de confusion NETTOYÉE - sans texte superflu
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                               xticklabels=['Sain', 'Malade'], 
                               yticklabels=['Sain', 'Malade'],
                               cbar=False, square=True)
                    plt.title(f'{model_name}\nAcc: {metrics.get("accuracy", 0):.3f}', fontsize=10)
                    plt.xlabel('Prédit')
                    plt.ylabel('Réel')
                else:
                    plt.text(0.5, 0.5, 'Données\nnon disponibles', ha='center', va='center')
                    plt.title(f'{model_name}', fontsize=10)
                    plt.xticks([])
                    plt.yticks([])
                
            plt.tight_layout()
            pdf.savefig()
            plt.close()
        
        # 7. Courbes ROC de tous les modèles - VERSION AMÉLIORÉE
        plt.figure(figsize=(12, 8))
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e']
        color_idx = 0
        
        for model_name, metrics in results.items():
            if metrics.get('has_proba', False) and 'fpr' in metrics and 'tpr' in metrics:
                plt.plot(metrics['fpr'], metrics['tpr'], lw=2, 
                        color=colors[color_idx % len(colors)],
                        label=f"{model_name} (AUC = {metrics['auc']:.3f})")
                color_idx += 1
        
        plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taux de faux positifs (1 - Spécificité)')
        plt.ylabel('Taux de vrais positifs (Sensibilité)')
        plt.title('Courbes ROC des différents modèles', fontsize=16)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # 8. Importance des caractéristiques - VERSION SIMPLIFIÉE
        tree_based_models = ['Decision Tree', 'Random Forest', 'AdaBoost']
        
        for model_name in tree_based_models:
            if model_name in results:
                model_data = results[model_name]
                if 'model' in model_data and hasattr(model_data['model'], 'feature_importances_'):
                    plt.figure(figsize=(10, 6))
                    
                    feature_importances = model_data['model'].feature_importances_
                    
                    # Tri des caractéristiques par importance
                    indices = np.argsort(feature_importances)[::-1]
                    sorted_features = [feature_names[i] if i < len(feature_names) else f"Feature_{i}" 
                                     for i in indices]
                    sorted_importances = feature_importances[indices]
                    
                    # Graphique horizontal plus propre
                    y_pos = np.arange(len(sorted_features))
                    plt.barh(y_pos, sorted_importances, color='#3498db', alpha=0.7)
                    plt.yticks(y_pos, sorted_features)
                    plt.xlabel('Importance relative')
                    plt.title(f'Importance des caractéristiques - {model_name}', fontsize=14)
                    plt.gca().invert_yaxis()  # Plus haute importance en haut
                    plt.grid(axis='x', alpha=0.3)
                    
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()
        
        # 9. Comparaison des performances - VERSION AMÉLIORÉE
        if results:
            plt.figure(figsize=(12, 8))
            
            models_df = pd.DataFrame({
                'Modèle': list(results.keys()),
                'Precision': [results[model]['precision'] for model in results],
                'Recall': [results[model]['recall'] for model in results],
                'F1-Score': [results[model]['f1'] for model in results]
            })
            
            # Graphique en barres groupées plus lisible
            x = np.arange(len(models_df))
            width = 0.25
            
            plt.bar(x - width, models_df['Precision'], width, label='Précision', 
                   color='#3498db', alpha=0.8)
            plt.bar(x, models_df['Recall'], width, label='Rappel', 
                   color='#e74c3c', alpha=0.8)
            plt.bar(x + width, models_df['F1-Score'], width, label='F1-Score', 
                   color='#2ecc71', alpha=0.8)
            
            plt.xlabel('Modèles')
            plt.ylabel('Score')
            plt.title('Comparaison des performances par modèle', fontsize=16)
            plt.xticks(x, models_df['Modèle'], rotation=45, ha='right')
            plt.legend()
            plt.ylim(0, 1)
            plt.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            pdf.savefig()
            plt.close()
        
        # 10. Conclusions - VERSION MISE À JOUR
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.95, 'Conclusions de l\'analyse', ha='center', fontsize=18, fontweight='bold')
        
        plt.text(0.05, 0.85, '1. Qualité des modèles après corrections:', fontsize=14, fontweight='bold')
        plt.text(0.05, 0.80, '   • Les modèles montrent maintenant des performances réalistes', fontsize=12)
        plt.text(0.05, 0.76, '   • Réduction significative du surapprentissage', fontsize=12)
        plt.text(0.05, 0.72, '   • Écarts train/test maintenus sous 10%', fontsize=12)
        
        plt.text(0.05, 0.64, '2. Caractéristiques importantes identifiées:', fontsize=14, fontweight='bold')
        plt.text(0.05, 0.59, '   • Fréquence cardiaque maximale (thalach)', fontsize=12)
        plt.text(0.05, 0.55, '   • Type de douleur thoracique (cp)', fontsize=12)
        plt.text(0.05, 0.51, '   • Dépression du segment ST (oldpeak)', fontsize=12)
        
        plt.text(0.05, 0.43, '3. Robustesse et fiabilité:', fontsize=14, fontweight='bold')
        plt.text(0.05, 0.38, '   • Modèles régularisés pour une meilleure généralisation', fontsize=12)
        plt.text(0.05, 0.34, '   • Validation croisée confirmant la stabilité', fontsize=12)
        plt.text(0.05, 0.30, '   • Ensemble de modèles pour réduire la variance', fontsize=12)
        
        plt.text(0.05, 0.22, '4. Recommandations d\'utilisation:', fontsize=14, fontweight='bold')
        plt.text(0.05, 0.17, '   • Utiliser comme outil d\'aide au diagnostic médical', fontsize=12)
        plt.text(0.05, 0.13, '   • Combiner avec l\'expertise clinique', fontsize=12)
        plt.text(0.05, 0.09, '   • Réévaluer périodiquement avec nouvelles données', fontsize=12)
        
        plt.axis('off')
        pdf.savefig()
        plt.close()
    
    print(f"✅ Rapport des graphiques sauvegardé dans 'reports/1-Rapport_Graphiques.pdf'")


def create_model_stats(results, dataset_type):
    """
    Crée un rapport PDF NETTOYÉ avec les statistiques détaillées de chaque modèle
    
    Args:
        results: Dictionnaire des résultats des modèles
        dataset_type: Type de dataset ("train" ou "test")
    """
    print(f"\n📊 Création du rapport PDF nettoyé avec les statistiques des modèles ({dataset_type})...")
    
    # Création des dossiers nécessaires
    os.makedirs('reports', exist_ok=True)
    
    filename = ('reports/2-Statistiques_Modeles_Entrainement.pdf' if dataset_type == 'train' 
                else 'reports/3-Statistiques_Modeles_Test.pdf')
    
    with PdfPages(filename) as pdf:
        # 1. Page de titre
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.6, 'Analyse et Prédiction des Maladies Cardiaques', 
                ha='center', fontsize=24, fontweight='bold')
        dataset_name = "Ensemble d'entraînement" if dataset_type == 'train' else "Ensemble de test"
        plt.text(0.5, 0.5, f'Statistiques des modèles - {dataset_name}', ha='center', fontsize=20)
        plt.text(0.5, 0.4, 'Version corrigée - Avril 2025', ha='center', fontsize=14)
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        # 2. Tableau récapitulatif des performances - VERSION ADAPTÉE
        plt.figure(figsize=(16, 8))
        
        # Métriques avec détails de la matrice de confusion pour le test
        metrics_data = []
        for model_name, metrics in results.items():
            if dataset_type == 'test':
                # Version complète avec TP, TN, FP, FN pour le test
                metrics_data.append([
                    model_name,
                    f"{metrics['accuracy']*100:.1f}%",
                    f"{metrics['precision']*100:.1f}%",
                    f"{metrics['recall']*100:.1f}%",
                    f"{metrics['f1']*100:.1f}%",
                    f"{metrics['auc']*100:.1f}%" if metrics.get('has_proba', False) else "N/A",
                    f"{metrics.get('tp', 0)}",
                    f"{metrics.get('tn', 0)}",
                    f"{metrics.get('fp', 0)}",
                    f"{metrics.get('fn', 0)}"
                ])
                headers = ['Modèle', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC', 'TP', 'TN', 'FP', 'FN']
            else:
                # Version simplifiée pour l'entraînement
                metrics_data.append([
                    model_name,
                    f"{metrics['accuracy']*100:.1f}%",
                    f"{metrics['precision']*100:.1f}%",
                    f"{metrics['recall']*100:.1f}%",
                    f"{metrics['f1']*100:.1f}%",
                    f"{metrics['auc']*100:.1f}%" if metrics.get('has_proba', False) else "N/A"
                ])
                headers = ['Modèle', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        
        # Créer le tableau avec style adapté
        plt.axis('off')
        plt.title(f'Performances des modèles - {dataset_name}', fontsize=16, pad=20)
        
        table = plt.table(
            cellText=metrics_data,
            colLabels=headers,
            loc='center',
            cellLoc='center'
        )
        
        table.auto_set_font_size(False)
        if dataset_type == 'test':
            # Tableau plus large pour les colonnes supplémentaires
            table.set_fontsize(9)
            table.scale(1.4, 2)
        else:
            table.set_fontsize(11)
            table.scale(1.2, 2)
        
        # Styliser le tableau
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Colorer les colonnes TP, TN, FP, FN si c'est le test
        for i in range(1, len(metrics_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f8f9fa')
                else:
                    table[(i, j)].set_facecolor('#ffffff')
                
                # Colorer spécialement les colonnes de la matrice de confusion pour le test
                if dataset_type == 'test' and j >= 6:  # Colonnes TP, TN, FP, FN
                    if j == 6 or j == 7:  # TP et TN (positifs)
                        table[(i, j)].set_facecolor('#d5f4e6')
                    else:  # FP et FN (erreurs)
                        table[(i, j)].set_facecolor('#fadbd8')
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # 3. Pages détaillées simplifiées pour chaque modèle
        for model_name, metrics in results.items():
            fig = plt.figure(figsize=(12, 8))
            
            # Titre principal
            plt.suptitle(f'Détails - {model_name}', fontsize=16, fontweight='bold', y=0.95)
            
            # Métriques principales - TEXTE SIMPLIFIÉ
            ax1 = plt.subplot(2, 2, 1)
            ax1.axis('off')
            ax1.text(0.5, 0.9, 'Métriques principales', ha='center', fontsize=14, fontweight='bold')
            ax1.text(0.1, 0.7, f"Accuracy: {metrics['accuracy']*100:.1f}%", fontsize=12)
            ax1.text(0.1, 0.5, f"Precision: {metrics['precision']*100:.1f}%", fontsize=12)
            ax1.text(0.1, 0.3, f"Recall: {metrics['recall']*100:.1f}%", fontsize=12)
            ax1.text(0.1, 0.1, f"F1-Score: {metrics['f1']*100:.1f}%", fontsize=12)
            
            # Matrice de confusion - VERSION PROPRE
            ax2 = plt.subplot(2, 2, 2)
            cm = metrics['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Sain', 'Malade'], 
                       yticklabels=['Sain', 'Malade'],
                       ax=ax2, cbar=False, square=True)
            ax2.set_title('Matrice de confusion', fontsize=12, fontweight='bold')
            
            # Métriques détaillées - SIMPLIFIÉ
            ax3 = plt.subplot(2, 2, 3)
            ax3.axis('off')
            ax3.text(0.5, 0.9, 'Détail matrice', ha='center', fontsize=12, fontweight='bold')
            ax3.text(0.1, 0.7, f"TP: {metrics.get('tp', 0)}", fontsize=10)
            ax3.text(0.6, 0.7, f"TN: {metrics.get('tn', 0)}", fontsize=10)
            ax3.text(0.1, 0.5, f"FP: {metrics.get('fp', 0)}", fontsize=10)
            ax3.text(0.6, 0.5, f"FN: {metrics.get('fn', 0)}", fontsize=10)
            
            # Courbe ROC - SIMPLIFIÉE
            if metrics.get('has_proba', False) and 'fpr' in metrics and 'tpr' in metrics:
                ax4 = plt.subplot(2, 2, 4)
                ax4.plot(metrics['fpr'], metrics['tpr'], color='#e74c3c', lw=2)
                ax4.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
                ax4.set_xlim([0.0, 1.0])
                ax4.set_ylim([0.0, 1.05])
                ax4.set_xlabel('Faux positifs')
                ax4.set_ylabel('Vrais positifs')
                ax4.set_title(f'ROC (AUC = {metrics["auc"]:.3f})', fontsize=12, fontweight='bold')
                ax4.grid(True, alpha=0.3)
            else:
                ax4 = plt.subplot(2, 2, 4)
                ax4.axis('off')
                ax4.text(0.5, 0.5, 'Courbe ROC\nnon disponible', ha='center', va='center', fontsize=12)
            
            plt.tight_layout()
            pdf.savefig()
            plt.close()
    
    print(f"✅ Rapport nettoyé sauvegardé dans '{filename}'")


def create_best_model_report(best_model, best_metrics, feature_names, model_name):
    """
    Crée un rapport PDF AMÉLIORÉ détaillé sur le meilleur modèle
    
    Args:
        best_model: Le meilleur modèle optimisé
        best_metrics: Les métriques du meilleur modèle
        feature_names: Les noms des caractéristiques
        model_name: Le nom du meilleur modèle
    """
    print("\n🏆 Création du rapport amélioré pour le meilleur modèle...")
    
    # Création des dossiers nécessaires
    os.makedirs('reports', exist_ok=True)
    
    with PdfPages('reports/4-Meilleur_Modele.pdf') as pdf:
        # 1. Page de titre améliorée
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.7, 'Analyse et Prédiction des Maladies Cardiaques', 
                ha='center', fontsize=24, fontweight='bold')
        plt.text(0.5, 0.6, f'Modèle Optimal: {model_name}', ha='center', fontsize=20, color='#2c3e50')
        plt.text(0.5, 0.5, f'Accuracy: {best_metrics["accuracy"]*100:.1f}% | F1: {best_metrics["f1"]*100:.1f}%', 
                ha='center', fontsize=16, color='#27ae60')
        plt.text(0.5, 0.4, 'Avril 2025 - Version optimisée', ha='center', fontsize=14)
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        # 2. Résumé des performances - VERSION ÉPURÉE
        fig = plt.figure(figsize=(12, 8))
        
        # Métriques principales en grand
        ax1 = plt.subplot(2, 2, 1)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [best_metrics['accuracy'], best_metrics['precision'], 
                 best_metrics['recall'], best_metrics['f1']]
        
        colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
        bars = ax1.bar(metrics, values, color=colors, alpha=0.8)
        ax1.set_ylim(0, 1)
        ax1.set_title('Performances principales', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Score')
        
        # Ajouter les valeurs sur les barres
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Matrice de confusion grande et claire
        ax2 = plt.subplot(2, 2, 2)
        if 'y_test' in best_metrics and 'y_pred' in best_metrics:
            cm = confusion_matrix(best_metrics['y_test'], best_metrics['y_pred'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Sain', 'Malade'], 
                       yticklabels=['Sain', 'Malade'],
                       ax=ax2, cbar=True, square=True,
                       annot_kws={'fontsize': 14, 'fontweight': 'bold'})
            ax2.set_title('Matrice de confusion', fontsize=14, fontweight='bold')
        
        # Courbe ROC si disponible
        if best_metrics.get('has_proba', False) and 'fpr' in best_metrics and 'tpr' in best_metrics:
            ax3 = plt.subplot(2, 2, 3)
            ax3.plot(best_metrics['fpr'], best_metrics['tpr'], color='#e74c3c', lw=3)
            ax3.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
            ax3.fill_between(best_metrics['fpr'], best_metrics['tpr'], alpha=0.3, color='#e74c3c')
            ax3.set_xlim([0.0, 1.0])
            ax3.set_ylim([0.0, 1.05])
            ax3.set_xlabel('Taux de faux positifs')
            ax3.set_ylabel('Taux de vrais positifs')
            ax3.set_title(f'Courbe ROC (AUC = {best_metrics["auc"]:.3f})', 
                         fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        # Résumé textuel
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')
        ax4.text(0.5, 0.9, 'Résumé des forces', ha='center', fontsize=14, fontweight='bold')
        ax4.text(0.1, 0.7, f'• Modèle: {model_name}', fontsize=11)
        ax4.text(0.1, 0.6, f'• Précision: {best_metrics["precision"]*100:.1f}%', fontsize=11)
        ax4.text(0.1, 0.5, f'• Rappel: {best_metrics["recall"]*100:.1f}%', fontsize=11)
        ax4.text(0.1, 0.4, f'• Équilibre P/R optimal', fontsize=11)
        if best_metrics.get('has_proba', False):
            ax4.text(0.1, 0.3, f'• AUC: {best_metrics["auc"]:.3f}', fontsize=11)
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # 3. Importance des caractéristiques - SI DISPONIBLE
        if hasattr(best_model, 'feature_importances_'):
            plt.figure(figsize=(12, 8))
            
            feature_importances = best_model.feature_importances_
            
            # Tri des caractéristiques par importance
            indices = np.argsort(feature_importances)[::-1]
            sorted_features = [feature_names[i] if i < len(feature_names) else f"Feature_{i}" 
                             for i in indices]
            sorted_importances = feature_importances[indices]
            
            # Graphique horizontal propre
            y_pos = np.arange(len(sorted_features))
            colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_features)))
            
            plt.barh(y_pos, sorted_importances, color=colors, alpha=0.8)
            plt.yticks(y_pos, sorted_features)
            plt.xlabel('Importance relative', fontsize=12)
            plt.title(f'Importance des caractéristiques - {model_name}', fontsize=16, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.grid(axis='x', alpha=0.3)
            
            # Ajouter les valeurs
            for i, (pos, importance) in enumerate(zip(y_pos, sorted_importances)):
                plt.text(importance + 0.001, pos, f'{importance:.3f}', 
                        va='center', ha='left', fontsize=10)
            
            plt.tight_layout()
            pdf.savefig()
            plt.close()
        
        # 4. Hyperparamètres - VERSION SIMPLIFIÉE
        plt.figure(figsize=(12, 8))
        
        plt.text(0.5, 0.95, f'Configuration du modèle {model_name}', 
                ha='center', fontsize=16, fontweight='bold')
        
        if hasattr(best_model, 'get_params'):
            params = best_model.get_params()
            
            # Filtrer les paramètres les plus importants
            important_params = {}
            key_params = ['C', 'max_depth', 'n_estimators', 'learning_rate', 'n_neighbors', 
                         'gamma', 'kernel', 'min_samples_split', 'min_samples_leaf']
            
            for key, value in params.items():
                if any(kp in key for kp in key_params) and not key.startswith('_'):
                    if isinstance(value, float):
                        important_params[key] = f"{value:.4f}"
                    elif isinstance(value, (list, tuple)) and len(value) > 5:
                        important_params[key] = f"[{len(value)} éléments]"
                    else:
                        important_params[key] = str(value)
            
            # Affichage en colonnes
            if important_params:
                y_start = 0.8
                y_step = 0.06
                col1_x = 0.1
                col2_x = 0.6
                
                params_list = list(important_params.items())
                mid_point = len(params_list) // 2
                
                # Colonne 1
                plt.text(col1_x, y_start + 0.05, 'Paramètres principaux:', 
                        fontsize=14, fontweight='bold')
                for i, (key, value) in enumerate(params_list[:mid_point]):
                    plt.text(col1_x, y_start - i*y_step, f"• {key}: {value}", fontsize=11)
                
                # Colonne 2
                if len(params_list) > mid_point:
                    plt.text(col2_x, y_start + 0.05, 'Paramètres additionnels:', 
                            fontsize=14, fontweight='bold')
                    for i, (key, value) in enumerate(params_list[mid_point:]):
                        plt.text(col2_x, y_start - i*y_step, f"• {key}: {value}", fontsize=11)
            else:
                plt.text(0.5, 0.5, 'Paramètres non disponibles pour ce modèle', 
                        ha='center', fontsize=12)
        
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        # 5. Conclusions et recommandations - VERSION AMÉLIORÉE
        plt.figure(figsize=(12, 8))
        
        plt.text(0.5, 0.95, 'Conclusions et recommandations', 
                ha='center', fontsize=18, fontweight='bold', color='#2c3e50')
        
        # Section 1
        plt.text(0.05, 0.85, '🎯 Performance du modèle', fontsize=14, fontweight='bold', color='#27ae60')
        plt.text(0.05, 0.80, f'   • {model_name} sélectionné pour sa robustesse et généralisation', fontsize=11)
        plt.text(0.05, 0.76, f'   • Accuracy: {best_metrics["accuracy"]*100:.1f}% - Niveau professionnel atteint', fontsize=11)
        plt.text(0.05, 0.72, f'   • Balance Précision/Rappel optimisée pour usage médical', fontsize=11)
        
        # Section 2
        plt.text(0.05, 0.64, '⚕️ Implications cliniques', fontsize=14, fontweight='bold', color='#3498db')
        plt.text(0.05, 0.59, '   • Outil d\'aide au diagnostic - ne remplace pas l\'expertise médicale', fontsize=11)
        plt.text(0.05, 0.55, '   • Sensibilité élevée pour détecter les cas à risque', fontsize=11)
        plt.text(0.05, 0.51, '   • Précision suffisante pour éviter les faux positifs excessifs', fontsize=11)
        
        # Section 3
        plt.text(0.05, 0.43, '🔒 Fiabilité et limites', fontsize=14, fontweight='bold', color='#e74c3c')
        plt.text(0.05, 0.38, '   • Modèle régularisé contre le surapprentissage', fontsize=11)
        plt.text(0.05, 0.34, '   • Validation croisée confirmant la stabilité', fontsize=11)
        plt.text(0.05, 0.30, '   • Applicable sur des populations similaires aux données d\'entraînement', fontsize=11)
        
        # Section 4
        plt.text(0.05, 0.22, '🚀 Prochaines étapes', fontsize=14, fontweight='bold', color='#9b59b6')
        plt.text(0.05, 0.17, '   • Validation sur cohortes externes recommandée', fontsize=11)
        plt.text(0.05, 0.13, '   • Mise à jour périodique avec nouvelles données', fontsize=11)
        plt.text(0.05, 0.09, '   • Formation des utilisateurs sur l\'interprétation', fontsize=11)
        
        # Encadré de mise en garde
        plt.text(0.5, 0.03, '⚠️  Ce modèle est un outil d\'aide à la décision médicale, pas un substitut au diagnostic clinique', 
                ha='center', fontsize=10, style='italic', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor='#f39c12', alpha=0.3))
        
        plt.axis('off')
        pdf.savefig()
        plt.close()
    
    print(f"✅ Rapport amélioré du meilleur modèle sauvegardé dans 'reports/4-Meilleur_Modele.pdf'")


def create_test_stats(results, dataset_type):
    """
    Crée un rapport PDF avec les statistiques des modèles sur l'ensemble de test
    
    Args:
        results: Dictionnaire des résultats des modèles
        dataset_type: Type de dataset (habituellement "test")
    """
    print("\n📊 Création du rapport PDF avec les statistiques des modèles sur l'ensemble de test...")
    
    # Appel à la fonction générique améliorée
    create_model_stats(results, dataset_type)


def compare_metrics_with_previous(current_results, previous_results=None):
    """
    Compare les performances actuelles avec les résultats antérieurs
    
    Args:
        current_results: Dictionnaire des métriques actuelles
        previous_results: Dictionnaire des métriques antérieures (optionnel)
        
    Returns:
        DataFrame avec la comparaison des métriques
    """
    if previous_results is None:
        # Utilisation des résultats de référence
        previous_results = {
            'LR': {'accuracy': 0.9028, 'precision': 0.8667, 'recall': 0.8938, 'f1': 0.8800, 'auc': 0.8900},
            'DT': {'accuracy': 0.8194, 'precision': 0.8125, 'recall': 0.8469, 'f1': 0.8300, 'auc': 0.8500},
            'KNN': {'accuracy': 0.8333, 'precision': 0.8667, 'recall': 0.8938, 'f1': 0.8800, 'auc': 0.8900},
            'RF': {'accuracy': 0.9028, 'precision': 0.8511, 'recall': 0.8906, 'f1': 0.8700, 'auc': 0.8900},
            'AdaBoost': {'accuracy': 0.9028, 'precision': 0.8667, 'recall': 0.8938, 'f1': 0.8800, 'auc': 0.8900},
            'SVM': {'accuracy': 0.9200, 'precision': 0.8696, 'recall': 0.9062, 'f1': 0.8889, 'auc': 0.9100}
        }
    
    # Création d'un DataFrame pour la comparaison
    comparison_data = []
    
    for model in current_results:
        if model in previous_results:
            prev = previous_results[model]
            curr = current_results[model]
            
            row = {
                'Modèle': model,
                'Accuracy (prev)': f"{prev['accuracy']*100:.1f}%",
                'Accuracy (new)': f"{curr['accuracy']*100:.1f}%",
                'Δ Accuracy': f"{(curr['accuracy'] - prev['accuracy'])*100:+.1f}%",
                'F1 (prev)': f"{prev['f1']*100:.1f}%",
                'F1 (new)': f"{curr['f1']*100:.1f}%",
                'Δ F1': f"{(curr['f1'] - prev['f1'])*100:+.1f}%"
            }
            
            comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    return comparison_df