import pandas as pd
from pathlib import Path
from informatiqueDecisionnelle.config import CLEVELAND_DATA_PATH, STATLOG_DATA_PATH

def load_datasets():
    """Charge les datasets Cleveland et Statlog"""
    try:
        cleveland_df = pd.read_csv(CLEVELAND_DATA_PATH)
        statlog_df = pd.read_csv(STATLOG_DATA_PATH)
        return cleveland_df, statlog_df
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Erreur lors du chargement des datasets: {e}")

def save_processed_data(df, filename):
    """Sauvegarde les données traitées"""
    processed_path = Path(__file__).parent.parent / "data" / "processed" / filename
    df.to_csv(processed_path, index=False)
    return processed_path