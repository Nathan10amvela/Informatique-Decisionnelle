from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def prepare_data(train_df, test_df=None, is_single_dataset=False):
    """Prépare les données pour l'entraînement et les tests"""
    print("\n🔄 Préparation des données pour le machine learning...")
    
    # Séparation des features et de la cible
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    
    if is_single_dataset:
        # Si on utilise un seul dataset, on fait une division train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
    elif test_df is not None:
        # Si on a un dataset de test séparé
        X_test = test_df.drop('target', axis=1)
        y_test = test_df['target']
    else:
        raise ValueError("Erreur: Dataset de test manquant ou paramètre is_single_dataset non spécifié")
    
    # Normalisation des caractéristiques
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✅ Dimensions des données après préparation:")
    print(f"- X_train_scaled: {X_train_scaled.shape}")
    print(f"- X_test_scaled: {X_test_scaled.shape}")
    print(f"- y_train: {y_train.shape}")
    print(f"- y_test: {y_test.shape}")
    
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler