import pandas as pd
import joblib
from pathlib import Path

def load_model(model_name):
    """Charge un modèle sauvegardé"""
    model_path = Path(__file__).parent.parent.parent / "models" / f"{model_name}.joblib"
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Modèle {model_name} non trouvé. Veuillez d'abord entraîner le modèle.")

def make_predictions(model_artifacts, new_data):
    """Fait des prédictions avec le modèle chargé"""
    model = model_artifacts['model']
    scaler = model_artifacts['scaler']
    feature_names = model_artifacts['feature_names']
    
    # Préparation des données
    if isinstance(new_data, pd.DataFrame):
        X_new = new_data
    else:
        X_new = pd.DataFrame(new_data, columns=feature_names)
    
    # Normalisation et prédiction
    X_new_scaled = scaler.transform(X_new)
    predictions = model.predict(X_new_scaled)
    
    # Résultats
    results = pd.DataFrame({
        'Patient': [f"Patient {i+1}" for i in range(len(X_new))],
        'Prédiction': ['Maladie cardiaque' if p == 1 else 'Pas de maladie cardiaque' for p in predictions]
    })
    
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_new_scaled)[:, 1]
        results['Probabilité (%)'] = [f"{p*100:.2f}%" for p in probabilities]
    else:
        results['Probabilité (%)'] = "Non disponible"
    
    return results