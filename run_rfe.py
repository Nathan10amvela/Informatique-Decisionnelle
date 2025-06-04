import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from informatiqueDecisionnelle.features import prepare_data

# -------------------------------
# 1) CHARGER ET PRÉPARER LES DONNÉES
# -------------------------------
# Remplace ces lignes par l’import / la lecture de ton CSV
train_df = pd.read_csv("data/raw/Heart_disease_statlog.csv") 
# test_df = pd.read_csv("chemin/vers/ton_test.csv")

# Pour l’exemple, on suppose que train_df et test_df sont déjà définis en DataFrame.
X_train, X_test, X_train_scaled_df, X_test_scaled_df, y_train, y_test, scaler = \
    prepare_data(
        train_df=train_df,
        test_df=None,
        is_single_dataset=True,
        feature_engineering=False,
        balance_classes=True,
        feature_selection=False
    )

# Affiche le nombre initial de features (doit être 13 dans ton cas)
print(f"➡️ Nombre initial de colonnes après préparation : {X_train_scaled_df.shape[1]}")

# -------------------------------
# 2) BOUCLE RFE
# -------------------------------
results = []  # On stockera des tuples (nombre_de_features, score)

# On part de 13 (ou X_train_scaled_df.shape[1]) et on va jusqu'à 1
for n_features in range(X_train_scaled_df.shape[1], 0, -1):
    # 2.1) Créer un estimateur (logistic regression ici)
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    
    # 2.2) Instancier RFE pour garder 'n_features' colonnes
    rfe = RFE(estimator=lr, n_features_to_select=n_features, step=1)
    
    # 2.3) Fit le RFE sur les données d’entraînement
    rfe.fit(X_train_scaled_df, y_train)
    
    # 2.4) Transformer X_train et X_test pour ne garder que les n_features sélectionnés
    X_train_rfe = rfe.transform(X_train_scaled_df)
    X_test_rfe  = rfe.transform(X_test_scaled_df)
    
    # 2.5) Évaluer la performance par validation croisée (ici 5-fold CV sur train)
    # On pourrait aussi entraîner puis scorer sur le jeu de test pour un résultat final.
    scores_cv = cross_val_score(
        lr, 
        X_train_rfe, 
        y_train, 
        cv=5, 
        scoring='accuracy'   # ou 'roc_auc', 'f1', etc., selon ta métrique
    )
    mean_score = scores_cv.mean()
    
    # 2.6) On enregistre (n_features, mean_score)
    results.append((n_features, mean_score))
    print(f"‣ Avec {n_features:2d} features => CV accuracy = {mean_score:.4f}")

# -------------------------------
# 3) AFFICHER / SAUVEGARDER LES RÉSULTATS
# -------------------------------
df_results = pd.DataFrame(results, columns=['n_features', 'cv_accuracy'])
df_results = df_results.sort_values(by='n_features', ascending=False)

print("\n📊 Résumé des performances :")
print(df_results.to_string(index=False))

# (Optionnel) Sauvegarde dans un CSV
df_results.to_csv("reports/rfe_results.csv", index=False)















import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import cross_val_score
import numpy as np
from informatiqueDecisionnelle.features import prepare_data

# Import optionnel pour GradientBoosting (si disponible)
try:
    from sklearn.ensemble import GradientBoostingClassifier
    GRADIENT_BOOSTING_AVAILABLE = True
except ImportError:
    GRADIENT_BOOSTING_AVAILABLE = False
    print("⚠️ GradientBoostingClassifier non disponible - ignoré")

try:
    from sklearn.svm import SVC
    SVM_AVAILABLE = True
except ImportError:
    SVM_AVAILABLE = False
    print("⚠️ SVC non disponible - ignoré")

# -------------------------------
# LES FONCTIONS (déjà définies dans l'artifact précédent)
# -------------------------------

def advanced_feature_selection(X_train_scaled_df, y_train, models_to_test=None):
    """Sélection de features en combinant plusieurs approches"""
    if models_to_test is None:
        models_to_test = {
            'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        }
    
    results = []
    feature_names = X_train_scaled_df.columns.tolist()
    
    print(f"🔍 Test avec {len(models_to_test)} modèles différents...")
    
    for n_features in range(X_train_scaled_df.shape[1], 0, -1):
        print(f"\n📊 Test avec {n_features} features:")
        
        iteration_results = []
        
        for model_name, model in models_to_test.items():
            rfe = RFE(estimator=model, n_features_to_select=n_features, step=1)
            rfe.fit(X_train_scaled_df, y_train)
            X_train_rfe = rfe.transform(X_train_scaled_df)
            
            scores_cv = cross_val_score(model, X_train_rfe, y_train, cv=5, scoring='accuracy')
            mean_score = scores_cv.mean()
            std_score = scores_cv.std()
            
            selected_features = [feature_names[i] for i, selected in enumerate(rfe.support_) if selected]
            
            iteration_results.append({
                'model': model_name,
                'n_features': n_features,
                'cv_accuracy': mean_score,
                'cv_std': std_score,
                'selected_features': selected_features
            })
            
            print(f"  {model_name:15s} => {mean_score:.4f} (±{std_score:.4f})")
        
        results.extend(iteration_results)
    
    return pd.DataFrame(results)

def statistical_feature_selection(X_train_scaled_df, y_train):
    """Sélection basée sur des tests statistiques"""
    print("\n🧮 Sélection par critères statistiques:")
    
    # Test F (ANOVA)
    f_selector = SelectKBest(score_func=f_classif, k='all')
    f_selector.fit(X_train_scaled_df, y_train)
    f_scores = f_selector.scores_
    
    # Information mutuelle
    mi_scores = mutual_info_classif(X_train_scaled_df, y_train, random_state=42)
    
    # Créer un DataFrame avec les scores
    feature_scores = pd.DataFrame({
        'feature': X_train_scaled_df.columns,
        'f_score': f_scores,
        'mutual_info': mi_scores
    })
    
    # Normaliser les scores pour les combiner
    feature_scores['f_score_norm'] = (feature_scores['f_score'] - feature_scores['f_score'].min()) / \
                                     (feature_scores['f_score'].max() - feature_scores['f_score'].min())
    feature_scores['mi_norm'] = (feature_scores['mutual_info'] - feature_scores['mutual_info'].min()) / \
                               (feature_scores['mutual_info'].max() - feature_scores['mutual_info'].min())
    
    # Score combiné
    feature_scores['combined_score'] = (feature_scores['f_score_norm'] + feature_scores['mi_norm']) / 2
    
    # Trier par importance
    feature_scores = feature_scores.sort_values('combined_score', ascending=False)
    
    return feature_scores

def find_best_configuration(results_df):
    """Trouve la meilleure configuration (modèle + nombre de features)"""
    # Grouper par modèle et trouver le meilleur score pour chaque
    best_by_model = results_df.loc[results_df.groupby('model')['cv_accuracy'].idxmax()]
    
    print("\n🏆 Meilleures configurations par modèle :")
    for _, row in best_by_model.iterrows():
        print(f"{row['model']:20s} : {row['n_features']:2d} features => {row['cv_accuracy']:.4f}")
        print(f"{'':22s} Features: {', '.join(row['selected_features'])}")
    
    # Configuration globalement meilleure
    best_overall = results_df.loc[results_df['cv_accuracy'].idxmax()]
    print(f"\n🥇 MEILLEURE CONFIGURATION GLOBALE :")
    print(f"Modèle: {best_overall['model']}")
    print(f"Nombre de features: {best_overall['n_features']}")
    print(f"Accuracy: {best_overall['cv_accuracy']:.4f} (±{best_overall['cv_std']:.4f})")
    print(f"Features sélectionnées: {', '.join(best_overall['selected_features'])}")
    
    return best_overall

# -------------------------------
# SCRIPT PRINCIPAL D'UTILISATION
# -------------------------------

def main():
    print("=" * 80)
    print("🚀 SÉLECTION AVANCÉE DE FEATURES")
    print("=" * 80)
    
    # 1) CHARGER ET PRÉPARER LES DONNÉES (comme dans ton code original)
    print("\n📂 Chargement des données...")
    train_df = pd.read_csv("data/raw/Heart_disease_statlog.csv") 
    
    X_train, X_test, X_train_scaled_df, X_test_scaled_df, y_train, y_test, scaler = \
        prepare_data(
            train_df=train_df,
            test_df=None,
            is_single_dataset=True,
            feature_engineering=False,
            balance_classes=True,
            feature_selection=False
        )
    
    print(f"✅ Données préparées : {X_train_scaled_df.shape[0]} échantillons, {X_train_scaled_df.shape[1]} features")
    print(f"Features disponibles : {list(X_train_scaled_df.columns)}")
    
    # 2) MÉTHODE 1 : ANALYSE STATISTIQUE DES FEATURES
    print("\n" + "="*50)
    print("📊 MÉTHODE 1 : ANALYSE STATISTIQUE")
    print("="*50)
    
    feature_importance = statistical_feature_selection(X_train_scaled_df, y_train)
    print("\n🔝 Top 10 des features les plus importantes :")
    print(feature_importance.head(10).to_string(index=False))
    
    # Sauvegarde optionnelle
    feature_importance.to_csv("reports/feature_importance_stats.csv", index=False)
    print("💾 Résultats sauvegardés dans 'reports/feature_importance_stats.csv'")
    
    # 3) MÉTHODE 2 : RFE AVEC PLUSIEURS MODÈLES
    print("\n" + "="*50)
    print("🤖 MÉTHODE 2 : RFE AVEC PLUSIEURS MODÈLES")
    print("="*50)
    
    # Définir les modèles à tester (selon disponibilité)
    models_to_test = {
        'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced'),
    }
    
    # Ajouter GradientBoosting si disponible
    if GRADIENT_BOOSTING_AVAILABLE:
        models_to_test['GradientBoosting'] = GradientBoostingClassifier(random_state=42)
    
    # Ajouter SVM si disponible (décommente la ligne suivante si tu veux l'inclure)
    # if SVM_AVAILABLE:
    #     models_to_test['SVM'] = SVC(random_state=42, class_weight='balanced')
    
    print(f"🔍 Modèles à tester : {list(models_to_test.keys())}")
    
    # Lancer l'analyse RFE
    results_df = advanced_feature_selection(X_train_scaled_df, y_train, models_to_test)
    
    # Sauvegarde des résultats détaillés
    results_df.to_csv("reports/rfe_detailed_results.csv", index=False)
    print("💾 Résultats détaillés sauvegardés dans 'reports/rfe_detailed_results.csv'")
    
    # 4) TROUVER LA MEILLEURE CONFIGURATION
    print("\n" + "="*50)
    print("🏆 ANALYSE DES RÉSULTATS")
    print("="*50)
    
    best_config = find_best_configuration(results_df)
    
    # 5) AFFICHAGE D'UN RÉSUMÉ PAR NOMBRE DE FEATURES
    print("\n📈 RÉSUMÉ PAR NOMBRE DE FEATURES :")
    summary = results_df.groupby('n_features').agg({
        'cv_accuracy': ['mean', 'max', 'std']
    }).round(4)
    summary.columns = ['Mean_Accuracy', 'Max_Accuracy', 'Std_Accuracy']
    print(summary.head(10).to_string())
    
    # 6) RECOMMANDATIONS
    print("\n" + "="*50)
    print("💡 RECOMMANDATIONS")
    print("="*50)
    
    # Trouver le nombre optimal de features (meilleur compromis performance/simplicité)
    best_accuracy = results_df['cv_accuracy'].max()
    threshold = best_accuracy - 0.01  # Tolérance de 1%
    
    optimal_configs = results_df[results_df['cv_accuracy'] >= threshold].sort_values('n_features')
    
    if not optimal_configs.empty:
        recommended = optimal_configs.iloc[0]  # Celui avec le moins de features
        print(f"\n🎯 CONFIGURATION RECOMMANDÉE (meilleur compromis) :")
        print(f"Modèle: {recommended['model']}")
        print(f"Nombre de features: {recommended['n_features']}")
        print(f"Accuracy: {recommended['cv_accuracy']:.4f}")
        print(f"Features: {', '.join(recommended['selected_features'])}")
    
    return results_df, feature_importance, best_config

# -------------------------------
# UTILISATION SIMPLE ET RAPIDE
# -------------------------------

def quick_analysis(X_train_scaled_df, y_train, top_n_features=5):
    """
    Version rapide pour tester seulement les N meilleures features
    """
    print(f"\n⚡ ANALYSE RAPIDE (top {top_n_features} features)")
    print("="*40)
    
    # 1) Obtenir les features les plus importantes
    feature_stats = statistical_feature_selection(X_train_scaled_df, y_train)
    top_features = feature_stats.head(top_n_features)['feature'].tolist()
    
    print(f"🔝 Top {top_n_features} features sélectionnées : {top_features}")
    
    # 2) Tester avec seulement ces features
    X_train_top = X_train_scaled_df[top_features]
    
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced'),
    }
    
    print(f"\n📊 Performance avec les {top_n_features} meilleures features :")
    for name, model in models.items():
        scores = cross_val_score(model, X_train_top, y_train, cv=5, scoring='accuracy')
        print(f"{name:20s} : {scores.mean():.4f} (±{scores.std():.4f})")
    
    return top_features

# -------------------------------
# EXEMPLES D'UTILISATION
# -------------------------------

if __name__ == "__main__":
    # UTILISATION COMPLÈTE
    print("🚀 Lancement de l'analyse complète...")
    results, importance, best = main()
    
    print("\n" + "="*80)
    print("✅ ANALYSE TERMINÉE !")
    print("="*80)
    
    # UTILISATION RAPIDE (optionnelle)
    # print("\n" + "="*80)
    # quick_analysis(X_train_scaled_df, y_train, top_n_features=7)