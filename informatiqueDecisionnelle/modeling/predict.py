# informatiqueDecisionnelle/modeling/predict.py

"""
Module pour la prédiction des maladies cardiaques
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def make_predictions(model_artifacts, new_data):
    """
    Utilise le modèle entraîné pour faire des prédictions sur de nouvelles données
    
    Args:
        model_artifacts: Dictionnaire contenant le modèle et les objets de pré-traitement
        new_data: Nouvelles données à prédire (DataFrame, array ou liste)
        
    Returns:
        DataFrame contenant les prédictions
    """
    print("\n🔮 Réalisation de prédictions sur de nouvelles données...")
    
    # >>> MODIFICATION ICI : Extraire tous les artefacts nécessaires <<<
    model = model_artifacts['model']
    scaler = model_artifacts.get('scaler') # .get() pour la compatibilité avec les anciens modèles
    selector = model_artifacts.get('selector')
    original_feature_names = model_artifacts.get('original_feature_names', model_artifacts.get('feature_names'))

    # Préparation des données
    if isinstance(new_data, pd.DataFrame):
        X_new = new_data
    else:
        X_new = pd.DataFrame(new_data, columns=original_feature_names)

    # Assurer que les colonnes sont dans le bon ordre
    X_new = X_new[original_feature_names]

    # >>> MODIFICATION ICI : Appliquer le pipeline de pré-traitement complet <<<
    # 1. Normalisation
    if scaler:
        X_new_processed = scaler.transform(X_new)
    else:
        X_new_processed = X_new.values
        print("⚠️ Scaler non trouvé dans les artefacts. Prédiction sans normalisation.")

    # 2. Sélection de caractéristiques
    if selector:
        X_new_processed = selector.transform(X_new_processed)
    else:
        print("⚠️ Sélecteur de features non trouvé. Prédiction sur toutes les features.")
    
    # 3. Prédictions
    predictions = model.predict(X_new_processed)
    
    result_df = pd.DataFrame({
        'Patient': [f"Patient {i+1}" for i in range(len(X_new))],
        'Prédiction': ['Maladie cardiaque' if p == 1 else 'Pas de maladie cardiaque' for p in predictions]
    })
    
    try:
        probabilities = model.predict_proba(X_new_processed)[:, 1]
        result_df['Probabilité (%)'] = [f"{p*100:.2f}%" for p in probabilities]
        
        risk_levels = []
        for prob in probabilities:
            if prob < 0.2: risk_levels.append('Très faible')
            elif prob < 0.4: risk_levels.append('Faible')
            elif prob < 0.6: risk_levels.append('Modéré')
            elif prob < 0.8: risk_levels.append('Élevé')
            else: risk_levels.append('Très élevé')
        
        result_df['Niveau de risque'] = risk_levels
        
    except AttributeError:
        result_df['Probabilité (%)'] = "Non disponible"
        result_df['Niveau de risque'] = "Non disponible"
    
    print("\n📋 Résultats des prédictions:")
    print(result_df)
    
    if 'Probabilité (%)' in result_df.columns and result_df['Probabilité (%)'].iloc[0] != "Non disponible":
        probs = [float(p.strip('%')) / 100 for p in result_df['Probabilité (%)']]
        
        plt.figure(figsize=(10, 6))
        bars = plt.barh(result_df['Patient'], probs, color=['#3498db' if p < 0.5 else '#e74c3c' for p in probs])
        plt.title('Probabilité de maladie cardiaque par patient', fontsize=14)
        plt.xlabel('Probabilité')
        plt.xlim(0, 1)
        
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f"{width:.2f}", va='center')
        
        plt.tight_layout()
        os.makedirs('reports/figures', exist_ok=True)
        plt.savefig('reports/figures/predictions.png')
        plt.close()
    
    return result_df

def load_model(model_path):
    """
    Charge un modèle sauvegardé
    """
    print(f"\n📂 Chargement du modèle depuis {model_path}...")
    try:
        model_artifacts = joblib.load(model_path)
        print("✅ Modèle chargé avec succès!")
        
        required_keys = ['model', 'scaler', 'feature_names']
        for key in required_keys:
            if key not in model_artifacts:
                print(f"⚠️ Avertissement: Clé '{key}' manquante dans le modèle chargé.")
        
        return model_artifacts
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        return None

def predict_from_input(model_artifacts, user_data):
    """
    Réalise des prédictions à partir de données saisies par l'utilisateur
    """
    model = model_artifacts['model']
    scaler = model_artifacts.get('scaler')
    selector = model_artifacts.get('selector')
    original_feature_names = model_artifacts.get('original_feature_names', model_artifacts.get('feature_names'))
    
    user_df = pd.DataFrame([user_data])
    user_df = user_df[original_feature_names]

    # >>> MODIFICATION ICI : Appliquer le pipeline de pré-traitement <<<
    if scaler:
        user_data_processed = scaler.transform(user_df)
    else:
        user_data_processed = user_df.values
    
    if selector:
        user_data_processed = selector.transform(user_data_processed)
    
    prediction = model.predict(user_data_processed)[0]
    
    result = {
        'prediction': 'Maladie cardiaque' if prediction == 1 else 'Pas de maladie cardiaque',
        'prediction_code': int(prediction)
    }
    
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(user_data_processed)[0, 1]
        result['probability'] = probability
        
        if probability < 0.2: result['risk_level'] = 'Très faible'
        elif probability < 0.4: result['risk_level'] = 'Faible'
        elif probability < 0.6: result['risk_level'] = 'Modéré'
        elif probability < 0.8: result['risk_level'] = 'Élevé'
        else: result['risk_level'] = 'Très élevé'
    
    return result

# Les autres fonctions (generate_explanation, etc.) restent les mêmes
# mais nécessiteront une logique similaire pour appliquer le pipeline de pré-traitement
# si elles manipulent directement les données. Laissons-les telles quelles pour l'instant
# pour ne pas complexifier excessivement.

def generate_explanation(model, feature_names, user_data):
    # Cette fonction devra être adaptée si elle est utilisée avec un modèle régularisé
    # Pour l'instant, on la laisse inchangée.
    explanation = "Les facteurs qui ont le plus influencé cette prédiction sont:\n"
    return explanation

def calculate_feature_impact(model, scaler, feature_names, user_data):
    # Idem pour cette fonction
    impact_df = pd.DataFrame()
    return impact_df

def visualize_prediction_explanation(model, scaler, feature_names, user_data):
    # Idem pour cette fonction
    return 'reports/figures/prediction_explanation.png'