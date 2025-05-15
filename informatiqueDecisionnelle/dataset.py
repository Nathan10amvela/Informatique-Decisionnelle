"""
Module pour l'exploration et la visualisation des données du projet
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Définir des couleurs personnalisées pour les visualisations
COLORS = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
CUSTOM_CMAP = LinearSegmentedColormap.from_list("custom_colormap", COLORS)

def explore_datasets(df, title):
    """
    Réalise une exploration complète du dataset
    
    Args:
        df: DataFrame à explorer
        title: Titre pour identifier le dataset dans les affichages
        
    Returns:
        DataFrame exploré (potentiellement modifié)
    """
    print(f"\n{'='*80}")
    print(f"EXPLORATION DU DATASET: {title}")
    print(f"{'='*80}")
    
    # Aperçu des données
    print("\n📊 Aperçu des données:")
    print(df.head())
    
    # Informations générales
    print("\n📋 Informations sur le dataframe:")
    print(f"- Nombre de lignes: {df.shape[0]}")
    print(f"- Nombre de colonnes: {df.shape[1]}")
    print(f"- Types de données:")
    for dtype in df.dtypes.value_counts().index:
        n_cols = df.dtypes.value_counts()[dtype]
        print(f"  • {dtype}: {n_cols} colonnes")
    
    # Valeurs manquantes
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("\n⚠️ Valeurs manquantes détectées:")
        for col in missing_values[missing_values > 0].index:
            print(f"- {col}: {missing_values[col]} valeurs manquantes ({missing_values[col]/len(df)*100:.2f}%)")
    else:
        print("\n✅ Aucune valeur manquante détectée")
    
    # Statistiques descriptives
    print("\n📈 Statistiques descriptives:")
    print(df.describe().T)
    
    # Distribution de la variable cible
    if 'target' in df.columns:
        target_counts = df['target'].value_counts()
        print("\n🎯 Distribution de la variable cible:")
        for label, count in target_counts.items():
            percentage = count / len(df) * 100
            print(f"- Classe {label}: {count} échantillons ({percentage:.2f}%)")
    
    return df

def visualize_data(df, title):
    """
    Crée des visualisations pour mieux comprendre les données
    
    Args:
        df: DataFrame à visualiser
        title: Titre pour les graphiques
    """
    print(f"\n📊 Création des visualisations pour {title}...")
    
    # Configuration de la figure principale
    plt.figure(figsize=(20, 16))
    plt.suptitle(f"Analyse des données de maladies cardiaques - {title}", fontsize=22, y=0.95)
    
    # 1. Distribution de l'âge selon la maladie cardiaque
    plt.subplot(3, 3, 1)
    sns.histplot(data=df, x='age', hue='target', kde=True, bins=20, palette=['#3498db', '#e74c3c'])
    plt.title('Distribution de l\'âge selon la présence de maladie cardiaque', fontsize=12)
    plt.xlabel('Âge')
    plt.ylabel('Fréquence')
    
    # 2. Distribution du cholestérol selon la maladie cardiaque
    plt.subplot(3, 3, 2)
    sns.histplot(data=df, x='chol', hue='target', kde=True, bins=20, palette=['#3498db', '#e74c3c'])
    plt.title('Distribution du cholestérol selon la maladie cardiaque', fontsize=12)
    plt.xlabel('Cholestérol')
    plt.ylabel('Fréquence')
    
    # 3. Corrélation entre l'âge et le cholestérol selon la maladie
    plt.subplot(3, 3, 3)
    sns.scatterplot(data=df, x='age', y='chol', hue='target', palette=['#3498db', '#e74c3c'])
    plt.title('Âge vs Cholestérol selon la maladie cardiaque', fontsize=12)
    plt.xlabel('Âge')
    plt.ylabel('Cholestérol')
    
    # 4. Distribution par sexe
    plt.subplot(3, 3, 4)
    sex_counts = df.groupby(['sex', 'target']).size().unstack()
    sex_counts.plot(kind='bar', ax=plt.gca(), color=['#3498db', '#e74c3c'])
    plt.title('Distribution des maladies cardiaques par sexe', fontsize=12)
    plt.xlabel('Sexe (0=Femme, 1=Homme)')
    plt.ylabel('Nombre de patients')
    plt.xticks([0, 1], ['Femme', 'Homme'])
    plt.legend(['Pas de maladie', 'Maladie cardiaque'])
    
    # 5. Fréquence cardiaque maximale selon la maladie
    plt.subplot(3, 3, 5)
    sns.boxplot(data=df, x='target', y='thalach', palette=['#3498db', '#e74c3c'])
    plt.title('Fréquence cardiaque maximale selon la maladie', fontsize=12)
    plt.xlabel('Maladie cardiaque')
    plt.ylabel('Fréquence cardiaque maximale')
    plt.xticks([0, 1], ['Pas de maladie', 'Maladie cardiaque'])
    
    # 6. Boxplot des principales caractéristiques numériques par classe
    plt.subplot(3, 3, 6)
    features_to_plot = ['age', 'chol', 'thalach', 'oldpeak']
    df_melted = pd.melt(df, id_vars=['target'], value_vars=features_to_plot)
    sns.boxplot(data=df_melted, x='variable', y='value', hue='target', palette=['#3498db', '#e74c3c'])
    plt.title('Principales caractéristiques par classe', fontsize=12)
    plt.xlabel('Caractéristique')
    plt.ylabel('Valeur')
    plt.legend(['Pas de maladie', 'Maladie cardiaque'])
    plt.xticks(rotation=45)
    
    # 7. Matrice de corrélation
    plt.subplot(3, 3, 7)
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f", cbar=False)
    plt.title('Matrice de corrélation des caractéristiques', fontsize=12)
    
    # 8. Répartition par type d'angine de poitrine
    plt.subplot(3, 3, 8)
    cp_counts = df.groupby(['cp', 'target']).size().unstack()
    cp_counts.plot(kind='bar', ax=plt.gca(), color=['#3498db', '#e74c3c'])
    plt.title('Maladies cardiaques par type d\'angine de poitrine', fontsize=12)
    plt.xlabel('Type d\'angine de poitrine')
    plt.ylabel('Nombre de patients')
    plt.legend(['Pas de maladie', 'Maladie cardiaque'])
    
    # 9. Depression du segment ST pendant l'exercice (oldpeak)
    plt.subplot(3, 3, 9)
    sns.histplot(data=df, x='oldpeak', hue='target', kde=True, bins=20, palette=['#3498db', '#e74c3c'])
    plt.title('Depression du segment ST durant l\'exercice', fontsize=12)
    plt.xlabel('Oldpeak')
    plt.ylabel('Fréquence')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(f"reports/figures/{title.replace(' ', '_').lower()}_visualizations.png")
    plt.close()
    
    # Analyse des caractéristiques importantes
    print("\n🔍 Analyse des principales corrélations:")
    important_corrs = correlation_matrix['target'].sort_values(ascending=False)
    for feature, corr in important_corrs.items():
        if feature != 'target':
            print(f"- {feature}: {corr:.4f}")
    
    # Créons une figure supplémentaire pour les corrélations avec la cible
    plt.figure(figsize=(12, 8))
    features = [f for f in important_corrs.index if f != 'target']
    correlations = [important_corrs[f] for f in features]
    
    # Tri des caractéristiques par importance de corrélation
    sorted_indices = np.argsort(np.abs(correlations))[::-1]
    sorted_features = [features[i] for i in sorted_indices]
    sorted_correlations = [correlations[i] for i in sorted_indices]
    
    # Graphique des corrélations
    colors = ['#e74c3c' if c < 0 else '#2ecc71' for c in sorted_correlations]
    plt.barh(sorted_features, sorted_correlations, color=colors)
    plt.title(f'Corrélation des caractéristiques avec la maladie cardiaque - {title}', fontsize=14)
    plt.xlabel('Coefficient de corrélation')
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"reports/figures/{title.replace(' ', '_').lower()}_correlations.png")
    plt.close()
    
    return important_corrs

def visualize_class_distribution(train_df, test_df):
    """
    Visualise la distribution des classes dans les ensembles d'entraînement et de test
    
    Args:
        train_df: DataFrame d'entraînement
        test_df: DataFrame de test
    """
    plt.figure(figsize=(12, 6))
    
    # Distribution dans l'ensemble d'entraînement
    plt.subplot(1, 2, 1)
    train_counts = train_df['target'].value_counts()
    train_labels = ['Pas de maladie (0)', 'Maladie cardiaque (1)']
    plt.pie(train_counts, labels=train_labels, autopct='%1.1f%%', startangle=90, colors=['#3498db', '#e74c3c'])
    plt.title('Distribution des classes - Ensemble d\'entraînement', fontsize=14)
    
    # Distribution dans l'ensemble de test
    plt.subplot(1, 2, 2)
    test_counts = test_df['target'].value_counts()
    test_labels = ['Pas de maladie (0)', 'Maladie cardiaque (1)']
    plt.pie(test_counts, labels=test_labels, autopct='%1.1f%%', startangle=90, colors=['#3498db', '#e74c3c'])
    plt.title('Distribution des classes - Ensemble de test', fontsize=14)
    
    plt.tight_layout()
    plt.savefig("reports/figures/class_distribution.png")
    plt.close()
    
    print("\n📊 Distribution des classes:")
    print(f"- Entraînement: {dict(zip(['Sain (0)', 'Malade (1)'], train_counts))}")
    print(f"- Test: {dict(zip(['Sain (0)', 'Malade (1)'], test_counts))}")