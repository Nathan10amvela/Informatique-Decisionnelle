"""
Module pour la prédiction des maladies cardiaques
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def make_predictions(model, scaler, new_data, feature_names):
    """
    Utilise le modèle entraîné pour faire des prédictions sur de nouvelles données
    
    Args:
        model: Le modèle entraîné
        scaler: Le scaler utilisé pour normaliser les données
        new_data: Nouvelles données à prédire
        feature_names: Noms des caractéristiques
        
    Returns:
        DataFrame contenant les prédictions
    """
    print("\n🔮 Réalisation de prédictions sur de nouvelles données...")
    
    # Préparation des données
    if isinstance(new_data, pd.DataFrame):
        # Si new_data est déjà un DataFrame
        X_new = new_data
    else:
        # Si new_data est un array ou une liste
        X_new = pd.DataFrame(new_data, columns=feature_names)
    
    # Normalisation
    X_new_scaled = scaler.transform(X_new)
    
    # Prédictions
    predictions = model.predict(X_new_scaled)
    
    # Ajout des probabilités si disponible
    result_df = pd.DataFrame({
        'Patient': [f"Patient {i+1}" for i in range(len(X_new))],
        'Prédiction': ['Maladie cardiaque' if p == 1 else 'Pas de maladie cardiaque' for p in predictions]
    })
    
    try:
        probabilities = model.predict_proba(X_new_scaled)[:, 1]
        result_df['Probabilité (%)'] = [f"{p*100:.2f}%" for p in probabilities]
        
        # Ajout d'une colonne pour le niveau de risque
        risk_levels = []
        for prob in probabilities:
            if prob < 0.2:
                risk_levels.append('Très faible')
            elif prob < 0.4:
                risk_levels.append('Faible')
            elif prob < 0.6:
                risk_levels.append('Modéré')
            elif prob < 0.8:
                risk_levels.append('Élevé')
            else:
                risk_levels.append('Très élevé')
        
        result_df['Niveau de risque'] = risk_levels
        
    except:
        result_df['Probabilité (%)'] = "Non disponible"
        result_df['Niveau de risque'] = "Non disponible"
    
    print("\n📋 Résultats des prédictions:")
    print(result_df)
    
    # Visualisation des prédictions
    if 'Probabilité (%)' in result_df.columns and result_df['Probabilité (%)'].iloc[0] != "Non disponible":
        # Conversion des probabilités en nombres
        probs = [float(p.strip('%')) / 100 for p in result_df['Probabilité (%)']]
        
        plt.figure(figsize=(10, 6))
        bars = plt.barh(result_df['Patient'], probs, color=['#3498db' if p < 0.5 else '#e74c3c' for p in probs])
        plt.title('Probabilité de maladie cardiaque par patient', fontsize=14)
        plt.xlabel('Probabilité')
        plt.xlim(0, 1)
        
        # Ajout des valeurs sur les barres
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
    
    Args:
        model_path: Chemin vers le fichier de modèle
        
    Returns:
        Dictionnaire contenant le modèle, le scaler et les noms des caractéristiques
    """
    print(f"\n📂 Chargement du modèle depuis {model_path}...")
    
    try:
        model_artifacts = joblib.load(model_path)
        print("✅ Modèle chargé avec succès!")
        
        # Vérification de la structure
        required_keys = ['model', 'scaler', 'feature_names']
        for key in required_keys:
            if key not in model_artifacts:
                print(f"⚠️ Avertissement: '{key}' manquant dans le modèle chargé")
        
        return model_artifacts
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        return None

def predict_from_input(model_artifacts, user_data):
    """
    Réalise des prédictions à partir de données saisies par l'utilisateur
    
    Args:
        model_artifacts: Dictionnaire contenant le modèle, le scaler et les noms des caractéristiques
        user_data: Dictionnaire contenant les valeurs des caractéristiques
        
    Returns:
        Résultat de la prédiction
    """
    # Extraction des composants du modèle
    model = model_artifacts['model']
    scaler = model_artifacts['scaler']
    feature_names = model_artifacts['feature_names']
    
    # Création d'un DataFrame à partir des données utilisateur
    user_df = pd.DataFrame([user_data])
    
    # Réorganisation des colonnes pour correspondre à l'ordre attendu
    if set(user_df.columns) != set(feature_names):
        missing_features = set(feature_names) - set(user_df.columns)
        extra_features = set(user_df.columns) - set(feature_names)
        
        if missing_features:
            print(f"⚠️ Caractéristiques manquantes: {missing_features}")
            # Ajout des caractéristiques manquantes avec des valeurs par défaut (0 ou None)
            for feature in missing_features:
                user_df[feature] = 0
        
        if extra_features:
            print(f"⚠️ Caractéristiques supplémentaires ignorées: {extra_features}")
            # Suppression des caractéristiques supplémentaires
            user_df = user_df.drop(columns=extra_features)
    
    # Réorganisation des colonnes dans le même ordre que feature_names
    user_df = user_df[feature_names]
    
    # Normalisation
    user_data_scaled = scaler.transform(user_df)
    
    # Prédiction
    prediction = model.predict(user_data_scaled)[0]
    
    result = {
        'prediction': 'Maladie cardiaque' if prediction == 1 else 'Pas de maladie cardiaque',
        'prediction_code': int(prediction)
    }
    
    # Ajout de la probabilité si disponible
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(user_data_scaled)[0, 1]
        result['probability'] = probability
        
        # Ajout du niveau de risque
        if probability < 0.2:
            result['risk_level'] = 'Très faible'
        elif probability < 0.4:
            result['risk_level'] = 'Faible'
        elif probability < 0.6:
            result['risk_level'] = 'Modéré'
        elif probability < 0.8:
            result['risk_level'] = 'Élevé'
        else:
            result['risk_level'] = 'Très élevé'
    
    return result

def generate_explanation(model, feature_names, user_data):
    """
    Génère une explication simple pour la prédiction
    
    Args:
        model: Modèle entraîné
        feature_names: Noms des caractéristiques
        user_data: Données de l'utilisateur
        
    Returns:
        Texte d'explication
    """
    explanation = "Les facteurs qui ont le plus influencé cette prédiction sont:\n"
    
    # Pour les modèles basés sur des arbres
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
        
        # Création d'un DataFrame avec les caractéristiques et leur importance
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances,
            'Value': user_data.values[0]
        })
        
        # Tri par importance
        importance_df = importance_df.sort_values('Importance', ascending=False).head(5)
        
        for _, row in importance_df.iterrows():
            explanation += f"- {row['Feature']}: {row['Value']} (importance: {row['Importance']:.4f})\n"
    
    # Pour les modèles linéaires
    elif hasattr(model, 'coef_'):
        coefficients = model.coef_[0]
        
        # Création d'un DataFrame avec les caractéristiques et leurs coefficients
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': np.abs(coefficients),
            'Direction': ['augmente' if c > 0 else 'diminue' for c in coefficients],
            'Value': user_data.values[0]
        })
        
        # Tri par importance (valeur absolue du coefficient)
        coef_df = coef_df.sort_values('Coefficient', ascending=False).head(5)
        
        for _, row in coef_df.iterrows():
            explanation += f"- {row['Feature']}: {row['Value']} ({row['Direction']} la probabilité, coefficient: {row['Coefficient']:.4f})\n"
    else:
        explanation += "Désolé, ce modèle ne fournit pas d'explication détaillée.\n"
    
    # Ajout de conseils généraux
    explanation += "\nConseils généraux basés sur les facteurs de risque des maladies cardiaques:\n"
    explanation += "- Maintenir une alimentation équilibrée pauvre en graisses saturées\n"
    explanation += "- Pratiquer une activité physique régulière\n"
    explanation += "- Éviter la consommation de tabac\n"
    explanation += "- Limiter la consommation d'alcool\n"
    explanation += "- Surveiller régulièrement sa tension artérielle et son cholestérol\n"
    
    return explanation

def calculate_feature_impact(model, scaler, feature_names, user_data):
    """
    Calcule l'impact de chaque caractéristique sur la prédiction
    
    Args:
        model: Modèle entraîné
        scaler: Scaler utilisé pour normaliser les données
        feature_names: Noms des caractéristiques
        user_data: Données de l'utilisateur
        
    Returns:
        DataFrame avec l'impact de chaque caractéristique
    """
    # Création d'un DataFrame à partir des données utilisateur
    user_df = pd.DataFrame([user_data])
    user_df = user_df[feature_names]  # Réorganisation des colonnes
    
    # Normalisation des données
    user_data_scaled = scaler.transform(user_df)
    
    # Prédiction de référence
    if hasattr(model, 'predict_proba'):
        baseline_prediction = model.predict_proba(user_data_scaled)[0, 1]
    else:
        baseline_prediction = model.predict(user_data_scaled)[0]
    
    # Calcul de l'impact de chaque caractéristique
    feature_impacts = []
    
    for i, feature in enumerate(feature_names):
        # Création d'un jeu de données modifié (valeur médiane pour cette caractéristique)
        modified_data = user_data_scaled.copy()
        # Utilisation de la valeur médiane (0 après normalisation)
        modified_data[0, i] = 0
        
        # Nouvelle prédiction
        if hasattr(model, 'predict_proba'):
            new_prediction = model.predict_proba(modified_data)[0, 1]
        else:
            new_prediction = model.predict(modified_data)[0]
        
        # Calcul de l'impact
        impact = baseline_prediction - new_prediction
        
        feature_impacts.append({
            'Feature': feature,
            'Value': user_df[feature].values[0],
            'Impact': impact,
            'Direction': 'Augmente' if impact > 0 else 'Diminue' if impact < 0 else 'Neutre'
        })
    
    # Création d'un DataFrame et tri par impact absolu
    impact_df = pd.DataFrame(feature_impacts)
    impact_df['Absolute Impact'] = np.abs(impact_df['Impact'])
    impact_df = impact_df.sort_values('Absolute Impact', ascending=False)
    
    return impact_df

def visualize_prediction_explanation(model, scaler, feature_names, user_data):
    """
    Crée une visualisation explicative de la prédiction
    
    Args:
        model: Modèle entraîné
        scaler: Scaler utilisé pour normaliser les données
        feature_names: Noms des caractéristiques
        user_data: Données de l'utilisateur
    """
    # Calcul de l'impact des caractéristiques
    impact_df = calculate_feature_impact(model, scaler, feature_names, user_data)
    
    # Création de la visualisation
    plt.figure(figsize=(12, 8))
    
    # Sélection des N caractéristiques les plus impactantes
    top_n = min(10, len(impact_df))
    top_features = impact_df.head(top_n)
    
    # Création du graphique
    bars = plt.barh(
        top_features['Feature'],
        top_features['Impact'],
        color=[('#e74c3c' if impact > 0 else '#3498db') for impact in top_features['Impact']]
    )
    
    # Ajout des valeurs
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 0.01 if width > 0 else width - 0.01
        ha = 'left' if width > 0 else 'right'
        plt.text(
            label_x_pos, bar.get_y() + bar.get_height()/2,
            f"{width:.4f}", va='center', ha=ha
        )
    
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.7)
    plt.title('Impact des caractéristiques sur la prédiction', fontsize=14)
    plt.xlabel('Impact sur la probabilité')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    os.makedirs('reports/figures', exist_ok=True)
    plt.savefig('reports/figures/prediction_explanation.png')
    plt.close()
    
    return 'reports/figures/prediction_explanation.png'