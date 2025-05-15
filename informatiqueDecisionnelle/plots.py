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
        plt.text(0.5, 0.5, 'Rapport des Visualisations', ha='center', fontsize=20)
        plt.text(0.5, 0.4, 'Date: Avril 2025', ha='center', fontsize=14)
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        # 2. Distribution des caractéristiques
        plt.figure(figsize=(15, 10))
        plt.suptitle('Distribution des caractéristiques principales', fontsize=16, y=0.95)
        
        features_to_plot = ['age', 'chol', 'thalach', 'oldpeak', 'trestbps']
        for i, feature in enumerate(features_to_plot):
            if i < 6:  # S'assurer de ne pas dépasser la taille de la grille
                plt.subplot(2, 3, i+1)
                sns.histplot(data=train_df, x=feature, hue='target', kde=True, palette=['#3498db', '#e74c3c'])
                plt.title(f'Distribution de {feature}')
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
        
        # 4. Distribution des maladies cardiaques par sexe
        plt.figure(figsize=(10, 6))
        sex_counts = train_df.groupby(['sex', 'target']).size().unstack()
        ax = sex_counts.plot(kind='bar', color=['#3498db', '#e74c3c'])
        plt.title('Distribution des maladies cardiaques par sexe', fontsize=16)
        plt.xlabel('Sexe (0=Femme, 1=Homme)')
        plt.ylabel('Nombre de patients')
        plt.xticks([0, 1], ['Femme', 'Homme'])
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
        
        features_to_plot = ['age', 'chol', 'thalach', 'oldpeak']
        for i, feature in enumerate(features_to_plot):
            plt.subplot(2, 2, i+1)
            sns.boxplot(x='target', y=feature, data=train_df, palette=['#3498db', '#e74c3c'])
            plt.title(f'{feature} par classe')
            plt.xlabel('Maladie cardiaque')
            plt.xticks([0, 1], ['Pas de maladie', 'Maladie cardiaque'])
            
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        
        num_models = len(results)

        num_rows = (num_models + 2) // 3  # Arrondir vers le haut pour avoir assez de lignes

        
        # 6. Matrice de confusion pour chaque modèle
        plt.figure(figsize=(15, 5*num_rows))
        plt.suptitle('Matrices de confusion des différents modèles', fontsize=16, y=0.95)
        




        for i, (model_name, metrics) in enumerate(results.items()):
  
        



            plt.subplot(num_rows, 3, i+1)
            cm = confusion_matrix(metrics['y_test'], metrics['y_pred'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Pas de maladie', 'Maladie'], 
                       yticklabels=['Pas de maladie', 'Maladie'])
            plt.title(f'Matrice de confusion - {model_name}')
            plt.xlabel('Prédit')
            plt.ylabel('Réel')
            
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # 7. Courbes ROC de tous les modèles
        plt.figure(figsize=(12, 8))
        
        for model_name, metrics in results.items():
            if metrics['has_proba']:
                plt.plot(metrics['fpr'], metrics['tpr'], lw=2, 
                        label=f"{model_name} (AUC = {metrics['auc']:.4f})")
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taux de faux positifs (1 - Spécificité)')
        plt.ylabel('Taux de vrais positifs (Sensibilité)')
        plt.title('Courbes ROC des différents modèles', fontsize=16)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # 8. Importance des caractéristiques pour les modèles basés sur des arbres
        tree_based_models = ['Decision Tree', 'Random Forest', 'AdaBoost']
        
        for model_name in tree_based_models:
            if model_name in results:
                plt.figure(figsize=(10, 6))
                
                if hasattr(results[model_name]['model'], 'feature_importances_'):
                    feature_importances = results[model_name]['model'].feature_importances_
                    features = feature_names
                    
                    # Tri des caractéristiques par importance
                    indices = np.argsort(feature_importances)[::-1]
                    sorted_features = [features[i] for i in indices]
                    sorted_importances = feature_importances[indices]
                    
                    # Affichage du graphique
                    plt.barh(range(len(sorted_features)), sorted_importances, align='center')
                    plt.yticks(range(len(sorted_features)), sorted_features)
                    plt.title(f'Importance des caractéristiques - {model_name}', fontsize=16)
                    plt.xlabel('Importance')
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()
        
        # 9. Précision vs Recall pour tous les modèles
        plt.figure(figsize=(10, 6))
        
        models_df = pd.DataFrame({
            'Modèle': list(results.keys()),
            'Precision': [results[model]['precision'] for model in results],
            'Recall': [results[model]['recall'] for model in results]
        })
        
        models_df_melted = pd.melt(models_df, id_vars=['Modèle'], value_vars=['Precision', 'Recall'])
        
        sns.barplot(x='Modèle', y='value', hue='variable', data=models_df_melted, palette=['#3498db', '#e74c3c'])
        plt.title('Precision vs Recall par modèle', fontsize=16)
        plt.ylim(0.5, 1.0)
        plt.xticks(rotation=45)
        plt.legend(title='Métrique')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # 10. Interprétation globale
        plt.figure(figsize=(12, 8))
        plt.text(0.1, 0.9, 'Interprétation des résultats', fontsize=18, fontweight='bold')
        plt.text(0.1, 0.8, '1. Les caractéristiques les plus discriminantes:', fontsize=14)
        plt.text(0.1, 0.76, '   - La fréquence cardiaque maximale (thalach) est généralement plus basse chez les patients malades', fontsize=12)
        plt.text(0.1, 0.72, '   - Le type d\'angine de poitrine (cp) est fortement corrélé avec la présence de maladie', fontsize=12)
        plt.text(0.1, 0.68, '   - La dépression du segment ST (oldpeak) est plus élevée chez les patients malades', fontsize=12)
        
        plt.text(0.1, 0.6, '2. Différences entre les sexes:', fontsize=14)
        plt.text(0.1, 0.56, '   - Les hommes présentent plus souvent des maladies cardiaques que les femmes', fontsize=12)
        plt.text(0.1, 0.52, '   - Le profil des symptômes peut varier selon le sexe', fontsize=12)
        
        plt.text(0.1, 0.44, '3. Performance des modèles:', fontsize=14)
        plt.text(0.1, 0.4, '   - Les modèles ensemble (Random Forest, AdaBoost) montrent généralement de meilleures performances', fontsize=12)
        plt.text(0.1, 0.36, '   - Le compromis Precision/Recall varie selon les modèles', fontsize=12)
        plt.text(0.1, 0.32, '   - L\'AUC est un bon indicateur de la capacité discriminante globale', fontsize=12)
        
        plt.text(0.1, 0.24, '4. Limites de l\'analyse:', fontsize=14)
        plt.text(0.1, 0.2, '   - Taille relativement limitée des échantillons', fontsize=12)
        plt.text(0.1, 0.16, '   - Certaines caractéristiques peuvent être interdépendantes', fontsize=12)
        plt.text(0.1, 0.12, '   - Risque de surapprentissage sur certains modèles (KNN notamment)', fontsize=12)
        
        plt.axis('off')
        pdf.savefig()
        plt.close()
    
    print(f"✅ Rapport des graphiques sauvegardé dans 'reports/1-Rapport_Graphiques.pdf'")

def create_model_stats(results, dataset_type):
    """
    Crée un rapport PDF avec les statistiques détaillées de chaque modèle
    
    Args:
        results: Dictionnaire des résultats des modèles
        dataset_type: Type de dataset ("train" ou "test")
    """
    print(f"\n📊 Création du rapport PDF avec les statistiques des modèles ({dataset_type})...")
    
    # Création des dossiers nécessaires
    os.makedirs('reports', exist_ok=True)
    
    filename = 'reports/2-Statistiques_Modeles_Entrainement.pdf' if dataset_type == 'train' else 'reports/3-Statistiques_Modeles_Test.pdf'
    
    with PdfPages(filename) as pdf:
        # 1. Page de titre
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.6, 'Analyse et Prédiction des Maladies Cardiaques', 
                ha='center', fontsize=24, fontweight='bold')
        dataset_name = "Ensemble d'entraînement" if dataset_type == 'train' else "Ensemble de test"
        plt.text(0.5, 0.5, f'Statistiques des modèles - {dataset_name}', ha='center', fontsize=20)
        plt.text(0.5, 0.4, 'Date: Avril 2025', ha='center', fontsize=14)
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        # 2. Tableau récapitulatif des performances
        plt.figure(figsize=(12, 4 + 0.5 * len(results)))
        
        # Création d'un DataFrame avec les métriques et les valeurs de la matrice de confusion
        metrics_df = pd.DataFrame({
            'Modèle': list(results.keys()),
            'Accuracy': [f"{results[model]['accuracy']*100:.2f}%" for model in results],
            'Precision': [f"{results[model]['precision']*100:.2f}%" for model in results],
            'Recall': [f"{results[model]['recall']*100:.2f}%" for model in results],
            'F1-Score': [f"{results[model]['f1']*100:.2f}%" for model in results],
            'AUC': [f"{results[model]['auc']*100:.2f}%" if results[model]['has_proba'] else "N/A" for model in results],
            'TP': [f"{results[model].get('tp', 0)}" for model in results],
            'TN': [f"{results[model].get('tn', 0)}" for model in results],
            'FP': [f"{results[model].get('fp', 0)}" for model in results],
            'FN': [f"{results[model].get('fn', 0)}" for model in results]
        })
        
        # Création d'un tableau
        plt.axis('off')
        plt.title(f'Performances des modèles sur l\'{dataset_name}', fontsize=16, y=1.05)
        
        # Création du tableau
        table = plt.table(
            cellText=metrics_df.values,
            colLabels=metrics_df.columns,
            loc='center',
            cellLoc='center',
            colColours=['#f1f1f1'] * len(metrics_df.columns),
            cellColours=[['#f9f9f9' if i % 2 == 0 else '#f1f1f1' for j in range(len(metrics_df.columns))] for i in range(len(metrics_df))]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # 3. Page détaillée pour chaque modèle
        for model_name, metrics in results.items():
            plt.figure(figsize=(12, 8))
            
            # Titre
            plt.text(0.5, 0.95, f'Statistiques détaillées - {model_name}', 
                    ha='center', fontsize=16, fontweight='bold')
            
            # Métriques principales
            plt.text(0.1, 0.85, 'Métriques principales:', fontsize=14, fontweight='bold')
            plt.text(0.1, 0.8, f"Accuracy: {metrics['accuracy']*100:.2f}%", fontsize=12)
            plt.text(0.1, 0.76, f"Precision: {metrics['precision']*100:.2f}%", fontsize=12)
            plt.text(0.1, 0.72, f"Recall (Sensibilité): {metrics['recall']*100:.2f}%", fontsize=12)
            plt.text(0.1, 0.68, f"F1-Score: {metrics['f1']*100:.2f}%", fontsize=12)
            
            if metrics['has_proba']:
                plt.text(0.1, 0.64, f"AUC: {metrics['auc']:.4f}", fontsize=12)
            
            # Matrice de confusion
            plt.text(0.1, 0.56, 'Matrice de confusion:', fontsize=14, fontweight='bold')
            
            # Création de la matrice de confusion
            ax1 = plt.axes([0.1, 0.25, 0.35, 0.25])
            cm = metrics['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Pas de maladie', 'Maladie'], 
                       yticklabels=['Pas de maladie', 'Maladie'],
                       ax=ax1)
            plt.title('Matrice de confusion')
            
            # Valeurs de la matrice de confusion
            plt.text(0.55, 0.56, 'Valeurs de la matrice de confusion:', fontsize=14, fontweight='bold')
            plt.text(0.55, 0.52, f"Vrais Positifs (TP): {metrics.get('tp', 0)}", fontsize=12)
            plt.text(0.55, 0.48, f"Vrais Négatifs (TN): {metrics.get('tn', 0)}", fontsize=12)
            plt.text(0.55, 0.44, f"Faux Positifs (FP): {metrics.get('fp', 0)}", fontsize=12)
            plt.text(0.55, 0.40, f"Faux Négatifs (FN): {metrics.get('fn', 0)}", fontsize=12)
            
            # Métriques par classe
            class_metrics = {
                'Healthy (0)': {
                    'Precision': metrics.get('precision_per_class', {}).get(0, 0),
                    'Recall': metrics.get('recall_per_class', {}).get(0, 0),
                    'F1-Score': metrics.get('f1_per_class', {}).get(0, 0)
                },
                'Sick (1)': {
                    'Precision': metrics.get('precision_per_class', {}).get(1, 0),
                    'Recall': metrics.get('recall_per_class', {}).get(1, 0),
                    'F1-Score': metrics.get('f1_per_class', {}).get(1, 0)
                }
            }
            
            plt.text(0.1, 0.15, 'Métriques par classe:', fontsize=14, fontweight='bold')
            
            # Healthy class
            plt.text(0.1, 0.10, 'Healthy (0):', fontsize=12, fontweight='bold')
            plt.text(0.1, 0.06, f"Precision: {class_metrics['Healthy (0)']['Precision']*100:.2f}%", fontsize=10)
            plt.text(0.1, 0.02, f"Recall: {class_metrics['Healthy (0)']['Recall']*100:.2f}%", fontsize=10)
            plt.text(0.3, 0.02, f"F1-Score: {class_metrics['Healthy (0)']['F1-Score']*100:.2f}%", fontsize=10)
            
            # Sick class
            plt.text(0.55, 0.10, 'Sick (1):', fontsize=12, fontweight='bold')
            plt.text(0.55, 0.06, f"Precision: {class_metrics['Sick (1)']['Precision']*100:.2f}%", fontsize=10)
            plt.text(0.55, 0.02, f"Recall: {class_metrics['Sick (1)']['Recall']*100:.2f}%", fontsize=10)
            plt.text(0.75, 0.02, f"F1-Score: {class_metrics['Sick (1)']['F1-Score']*100:.2f}%", fontsize=10)
            
            # Courbe ROC si disponible
            if metrics['has_proba']:
                plt.text(0.1, 0.36, 'Courbe ROC:', fontsize=14, fontweight='bold')
                
                # Création de la courbe ROC
                ax2 = plt.axes([0.55, 0.20, 0.35, 0.15])
                plt.plot(metrics['fpr'], metrics['tpr'], color='#3498db', lw=2, 
                        label=f"AUC = {metrics['auc']:.4f}")
                plt.plot([0, 1], [0, 1], 'k--', lw=1)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('Taux de faux positifs')
                plt.ylabel('Taux de vrais positifs')
                plt.legend(loc='lower right', fontsize=8)
                plt.title('Courbe ROC')
            
            plt.axis('off')
            pdf.savefig()
            plt.close()
    
    print(f"✅ Rapport des statistiques sauvegardé dans '{filename}'")
    
def create_test_stats(results, dataset_type):
    """
    Crée un rapport PDF avec les statistiques des modèles sur l'ensemble de test
    
    Args:
        results: Dictionnaire des résultats des modèles
        dataset_type: Type de dataset (habituellement "test")
    """
    print("\n📊 Création du rapport PDF avec les statistiques des modèles sur l'ensemble de test...")
    
    # Appel à la fonction générique
    create_model_stats(results, dataset_type)

def create_best_model_report(best_model, best_metrics, feature_names, model_name):
    """
    Crée un rapport PDF détaillé sur le meilleur modèle
    
    Args:
        best_model: Le meilleur modèle optimisé
        best_metrics: Les métriques du meilleur modèle
        feature_names: Les noms des caractéristiques
        model_name: Le nom du meilleur modèle
    """
    print("\n🏆 Création du rapport détaillé pour le meilleur modèle...")
    
    # Création des dossiers nécessaires
    os.makedirs('reports', exist_ok=True)
    
    with PdfPages('reports/4-Meilleur_Modele.pdf') as pdf:
        # 1. Page de titre
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.6, 'Analyse et Prédiction des Maladies Cardiaques', 
                ha='center', fontsize=24, fontweight='bold')
        plt.text(0.5, 0.5, f'Modèle Optimal: {model_name}', ha='center', fontsize=20)
        plt.text(0.5, 0.4, 'Date: Avril 2025', ha='center', fontsize=14)
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        # 2. Résumé des performances
        plt.figure(figsize=(12, 8))
        
        plt.text(0.5, 0.95, 'Résumé des performances', ha='center', fontsize=16, fontweight='bold')
        
        # Comparaison avec les résultats antérieurs
        plt.text(0.1, 0.85, 'Comparaison avec les résultats antérieurs:', fontsize=14, fontweight='bold')
        
        # Création d'un tableau de comparaison (à adapter selon vos données antérieures)
        previous_results = {
            'Accuracy': 0.9200,
            'Precision': 0.8696,
            'Recall': 0.9062,
            'F1-Score': 0.8889,
            'AUC': 0.9100
        }
        
        current_results = {
            'Accuracy': best_metrics['accuracy'],
            'Precision': best_metrics['precision'],
            'Recall': best_metrics['recall'],
            'F1-Score': best_metrics['f1'],
            'AUC': best_metrics['auc'] if best_metrics['has_proba'] else 0
        }
        
        # Calcul des différences
        differences = {
            key: (current_results[key] - previous_results[key]) * 100 
            for key in previous_results.keys()
        }
        
        # Création du tableau de comparaison
        comparison_data = {
            'Métrique': list(previous_results.keys()),
            'Résultats antérieurs': [f"{value*100:.2f}%" for value in previous_results.values()],
            'Résultats actuels': [f"{value*100:.2f}%" for value in current_results.values()],
            'Différence': [f"{diff:.2f}%" + (" ↑" if diff > 0 else " ↓" if diff < 0 else "") for diff in differences.values()]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Affichage du tableau
        ax = plt.axes([0.1, 0.5, 0.8, 0.3])
        ax.axis('off')
        table = plt.table(
            cellText=comparison_df.values,
            colLabels=comparison_df.columns,
            loc='center',
            cellLoc='center',
            colColours=['#f1f1f1'] * len(comparison_df.columns),
            cellColours=[['#f9f9f9' if i % 2 == 0 else '#f1f1f1' for j in range(len(comparison_df.columns))] for i in range(len(comparison_df))]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Matrice de confusion
        plt.text(0.1, 0.45, 'Matrice de confusion:', fontsize=14, fontweight='bold')
        
        ax1 = plt.axes([0.1, 0.15, 0.35, 0.25])
        cm = confusion_matrix(best_metrics['y_test'], best_metrics['y_pred'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Pas de maladie', 'Maladie'], 
                   yticklabels=['Pas de maladie', 'Maladie'],
                   ax=ax1)
        plt.title('Matrice de confusion')
        
        # Courbe ROC
        if best_metrics['has_proba']:
            plt.text(0.55, 0.45, 'Courbe ROC:', fontsize=14, fontweight='bold')
            
            ax2 = plt.axes([0.55, 0.15, 0.35, 0.25])
            plt.plot(best_metrics['fpr'], best_metrics['tpr'], color='#3498db', lw=2, 
                    label=f"AUC = {best_metrics['auc']:.4f}")
            plt.plot([0, 1], [0, 1], 'k--', lw=1)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Taux de faux positifs')
            plt.ylabel('Taux de vrais positifs')
            plt.legend(loc='lower right', fontsize=8)
            plt.title('Courbe ROC')
        
        pdf.savefig()
        plt.close()
        
        # 3. Importance des caractéristiques (si disponible)
        if hasattr(best_model, 'feature_importances_'):
            plt.figure(figsize=(12, 8))
            
            feature_importances = best_model.feature_importances_
            features = feature_names
            
            # Tri des caractéristiques par importance
            indices = np.argsort(feature_importances)[::-1]
            sorted_features = [features[i] for i in indices]
            sorted_importances = feature_importances[indices]
            
            # Affichage du graphique
            plt.barh(range(len(sorted_features)), sorted_importances, align='center')
            plt.yticks(range(len(sorted_features)), sorted_features)
            plt.title('Importance des caractéristiques', fontsize=16)
            plt.xlabel('Importance')
            
            plt.tight_layout()
            pdf.savefig()
            plt.close()
        
        # 4. Courbe de précision-rappel
        if best_metrics['has_proba']:
            plt.figure(figsize=(12, 8))
            
            # Calcul de la courbe précision-rappel
            y_proba = best_metrics.get('y_proba', None)
            if y_proba is not None:
                precision, recall, _ = precision_recall_curve(best_metrics['y_test'], y_proba)
                
                plt.plot(recall, precision, lw=2)
                plt.xlabel('Recall (Sensibilité)')
                plt.ylabel('Precision')
                plt.title('Courbe Precision-Recall', fontsize=16)
                plt.grid(True, alpha=0.3)
                
                # Ajout du score moyen de précision (AP)
                average_precision = np.mean(precision)
                plt.text(0.5, 0.5, f'AP = {average_precision:.4f}', fontsize=12,
                       bbox=dict(facecolor='white', alpha=0.8))
                
                plt.tight_layout()
                pdf.savefig()
                plt.close()
        
        # 5. Hyperparamètres optimaux
        plt.figure(figsize=(12, 8))
        
        plt.text(0.5, 0.95, 'Hyperparamètres optimaux', ha='center', fontsize=16, fontweight='bold')
        
        # Récupération des hyperparamètres du modèle
        if hasattr(best_model, 'get_params'):
            params = best_model.get_params()
            
            # Filtrage des paramètres importants (à adapter selon le modèle)
            important_params = {}
            for key, value in params.items():
                # Exclusion des paramètres moins importants ou redondants
                if key.startswith('_') or key in ['base_estimator_', 'estimators_']:
                    continue
                
                # Formatage des valeurs pour une meilleure lisibilité
                if isinstance(value, float):
                    important_params[key] = f"{value:.4f}"
                elif isinstance(value, (list, tuple)) and len(value) > 10:
                    important_params[key] = f"{str(value[:3])[:-1]}, ..., {str(value[-3:])[1:]}"
                else:
                    important_params[key] = str(value)
            
            # Création du tableau des hyperparamètres
            params_df = pd.DataFrame({
                'Paramètre': list(important_params.keys()),
                'Valeur': list(important_params.values())
            })
            
            # Ajustement de la disposition selon le nombre de paramètres
            table_height = min(0.8, 0.1 * len(params_df))
            ax = plt.axes([0.1, 0.1, 0.8, table_height])
            ax.axis('off')
            
            table = plt.table(
                cellText=params_df.values,
                colLabels=params_df.columns,
                loc='center',
                cellLoc='left',
                colColours=['#f1f1f1'] * 2,
                cellColours=[['#f9f9f9' if i % 2 == 0 else '#f1f1f1' for j in range(2)] for i in range(len(params_df))]
            )
            
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            
        else:
            plt.text(0.5, 0.5, 'Paramètres non disponibles pour ce modèle', ha='center', fontsize=12)
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # 6. Conclusions et recommandations
        plt.figure(figsize=(12, 8))
        
        plt.text(0.5, 0.95, 'Conclusions et recommandations', ha='center', fontsize=16, fontweight='bold')
        
        plt.text(0.1, 0.85, '1. Résumé des performances:', fontsize=14, fontweight='bold')
        plt.text(0.1, 0.8, f"   - Le modèle {model_name} offre les meilleures performances globales", fontsize=12)
        plt.text(0.1, 0.76, f"   - Accuracy: {current_results['Accuracy']*100:.2f}%, Recall: {current_results['Recall']*100:.2f}%, Precision: {current_results['Precision']*100:.2f}%", fontsize=12)
        plt.text(0.1, 0.72, f"   - L'amélioration par rapport aux résultats antérieurs est de {differences['F1-Score']:.2f}% sur le F1-Score", fontsize=12)
        
        plt.text(0.1, 0.64, '2. Points forts du modèle:', fontsize=14, fontweight='bold')
        plt.text(0.1, 0.6, "   - Équilibre optimal entre précision et rappel", fontsize=12)
        plt.text(0.1, 0.56, "   - Robustesse face aux données déséquilibrées", fontsize=12)
        plt.text(0.1, 0.52, "   - Bonne capacité de généralisation sur des données inconnues", fontsize=12)
        
        plt.text(0.1, 0.44, '3. Limites et axes d\'amélioration:', fontsize=14, fontweight='bold')
        plt.text(0.1, 0.4, "   - Tester des ensembles de modèles plus complexes", fontsize=12)
        plt.text(0.1, 0.36, "   - Collecter davantage de données pour améliorer la robustesse", fontsize=12)
        plt.text(0.1, 0.32, "   - Explorer d'autres caractéristiques/biomarqueurs", fontsize=12)
        
        plt.text(0.1, 0.24, '4. Recommandations cliniques:', fontsize=14, fontweight='bold')
        plt.text(0.1, 0.2, "   - Utiliser ce modèle comme outil d'aide à la décision, pas comme substitut au diagnostic médical", fontsize=12)
        plt.text(0.1, 0.16, "   - Porter une attention particulière aux patients présentant des caractéristiques clés identifiées", fontsize=12)
        plt.text(0.1, 0.12, "   - Évaluer périodiquement la pertinence du modèle à mesure que de nouvelles données cliniques sont disponibles", fontsize=12)
        
        plt.axis('off')
        pdf.savefig()
        plt.close()
    
    print(f"✅ Rapport du meilleur modèle sauvegardé dans 'reports/4-Meilleur_Modele.pdf'")

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
        # Utilisation des résultats de l'étude précédente (à adapter selon vos données)
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
                'Accuracy (prev)': f"{prev['accuracy']*100:.2f}%",
                'Accuracy (new)': f"{curr['accuracy']*100:.2f}%",
                'Δ Accuracy': f"{(curr['accuracy'] - prev['accuracy'])*100:+.2f}%",
                'Recall (prev)': f"{prev['recall']*100:.2f}%",
                'Recall (new)': f"{curr['recall']*100:.2f}%",
                'Δ Recall': f"{(curr['recall'] - prev['recall'])*100:+.2f}%",
                'Precision (prev)': f"{prev['precision']*100:.2f}%",
                'Precision (new)': f"{curr['precision']*100:.2f}%",
                'Δ Precision': f"{(curr['precision'] - prev['precision'])*100:+.2f}%",
                'F1 (prev)': f"{prev['f1']*100:.2f}%",
                'F1 (new)': f"{curr['f1']*100:.2f}%",
                'Δ F1': f"{(curr['f1'] - prev['f1'])*100:+.2f}%"
            }
            
            comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    return comparison_df