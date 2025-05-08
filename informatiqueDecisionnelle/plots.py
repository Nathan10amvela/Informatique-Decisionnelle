import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from informatiqueDecisionnelle.config import PLOT_PARAMS

# Configuration des styles
plt.style.use(PLOT_PARAMS['style'])
colors = PLOT_PARAMS['colors']

def plot_data_exploration(df, title):
    """Crée des visualisations pour l'exploration des données"""
    print(f"\n📊 Création des visualisations pour {title}...")
    
    # Configuration de la figure principale
    plt.figure(figsize=PLOT_PARAMS['figsize'])
    plt.suptitle(f"Analyse des données de maladies cardiaques - {title}", fontsize=22, y=0.95)
    
    # 1. Distribution de l'âge selon la maladie cardiaque
    plt.subplot(3, 3, 1)
    sns.histplot(data=df, x='age', hue='target', kde=True, bins=20, palette=[colors[0], colors[1]])
    plt.title('Distribution de l\'âge selon la présence de maladie cardiaque', fontsize=12)
    plt.xlabel('Âge')
    plt.ylabel('Fréquence')
    
    # ... (autres visualisations comme dans votre code original)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def plot_model_performance(results):
    """Affiche les performances des modèles"""
    # Création d'un DataFrame pour la comparaison
    models_df = pd.DataFrame({
        'Modèle': list(results.keys()),
        'Accuracy': [results[model]['accuracy'] for model in results],
        'Precision': [results[model]['precision'] for model in results],
        'Recall': [results[model]['recall'] for model in results],
        'F1-Score': [results[model]['f1'] for model in results],
        'AUC': [results[model]['auc'] for model in results if results[model]['has_proba']]
    })
    
    # Visualisation des métriques
    plt.figure(figsize=(15, 12))
    
    # 1. Comparaison des accuracies
    plt.subplot(2, 2, 1)
    sns.barplot(x='Modèle', y='Accuracy', data=models_df, palette=colors)
    plt.title('Accuracy par modèle', fontsize=14)
    plt.ylim(0.5, 1.0)
    plt.xticks(rotation=45)
    
    # ... (autres visualisations de performance comme dans votre code original)
    
    plt.tight_layout()
    plt.show()
    
    return models_df.sort_values('F1-Score', ascending=False)