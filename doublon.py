import pandas as pd
import numpy as np

# Chargement des datasets
cleveland_df = pd.read_csv('Heart_disease_cleveland_new.csv')
statlog_df = pd.read_csv('Heart_disease_statlog.csv')

# Création d'une "empreinte" pour chaque échantillon
def create_fingerprint(df):
    return df.astype(str).sum(axis=1)

cleveland_fingerprints = create_fingerprint(cleveland_df)
statlog_fingerprints = create_fingerprint(statlog_df)

# Recherche des correspondances
matches = set(cleveland_fingerprints).intersection(set(statlog_fingerprints))
overlap_count = len(matches)
overlap_percentage = (overlap_count / len(statlog_df)) * 100

print(f"Nombre d'échantillons dans Cleveland: {len(cleveland_df)}")
print(f"Nombre d'échantillons dans Statlog: {len(statlog_df)}")
print(f"Nombre d'échantillons en commun: {overlap_count}")
print(f"Pourcentage de Statlog qui apparaît aussi dans Cleveland: {overlap_percentage:.2f}%")
