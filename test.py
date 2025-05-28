import pandas as pd

def find_common_rows(df1, df2):

    # Fusionner les dataframes pour trouver les intersections
    common_rows = pd.merge(df1, df2, how='inner')
    
    return common_rows, len(common_rows)

# Exemple d'utilisation
if __name__ == "__main__":
    df1 = pd.read_csv('data/raw/Heart_disease_statlog.csv')  
    df2 = pd.read_csv('data/raw/Heart_disease_cleveland_new.csv')
    
    # Trouver les lignes communes
    common, count = find_common_rows(df1, df2)
    
    print(f"Nombre de lignes communes: {count}")
    print("\nLignes communes:")
    print(common)