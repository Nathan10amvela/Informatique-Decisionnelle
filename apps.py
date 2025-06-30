import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import base64
from io import BytesIO

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de Maladies Cardiaques",
    page_icon="🫀",
    layout="wide"
)

# Titre de l'application
st.title('🫀 Prédiction de Maladies Cardiaques')
st.markdown("""
Cette application utilise le machine learning pour prédire le risque de maladie cardiaque 
en se basant sur des indicateurs médicaux. Entrez les informations du patient dans le panneau 
latéral pour obtenir une prédiction.
""")

# Création d'onglets
tab1, tab2, tab3 = st.tabs(["Prédiction", "À propos du modèle", "Statistiques"])

# Fonction pour charger le modèle
@st.cache_resource
def load_model(model_path='heart_disease_random_forest.joblib'):
    try:
        model_artifacts = joblib.load(model_path)
        return model_artifacts
    except:
        return None

# Fonction pour créer un téléchargement pour une figure
def get_figure_download_link(fig, filename="figure.png", text="Télécharger la figure"):
    """Génère un lien pour télécharger une figure matplotlib"""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Section pour la sélection du modèle
# NOUVEAU CODE CORRIGÉ
with st.sidebar:
    st.header('Modèle')
    model_options = {
        'Forêt Aléatoire (Standard)': 'models/heart_disease_random_forest.joblib',
        'Ensemble Régularisé (Recommandé)': 'models/heart_disease_voting_ensemble_regularized.joblib'
    }
    selected_model_name = st.selectbox('Sélectionnez un modèle:', list(model_options.keys()))
    selected_model_path = model_options[selected_model_name]

    # Chargement du modèle
    model_artifacts = load_model(selected_model_path)
    
    # Initialisation des variables pour éviter les erreurs
    model, scaler, selector, feature_names, original_feature_names = (None, None, None, None, None)
    
    if model_artifacts:
        st.success(f"Modèle '{selected_model_name}' chargé avec succès!")
        # Dépaquetage sécurisé des artefacts
        model = model_artifacts.get('model')
        scaler = model_artifacts.get('scaler')
        selector = model_artifacts.get('selector') # Pour les modèles régularisés
        feature_names = model_artifacts.get('feature_names') # Noms après sélection
        original_feature_names = model_artifacts.get('original_feature_names', feature_names) # Noms avant sélection
        date_created = model_artifacts.get('date_created', 'Non spécifiée')
        st.info(f"Date de création: {date_created}")
        
        if selector:
            st.info("Ce modèle utilise une sélection de caractéristiques.")
    else:
        st.error(f"Erreur: Impossible de charger le modèle depuis '{selected_model_path}'")
        st.info("Assurez-vous que le modèle se trouve dans le dossier 'models'.")

# Sidebar pour les informations du patient
with st.sidebar:
    st.header('Informations du patient')
    
    # Création des entrées pour chaque caractéristique
    def user_input_features():
        age = st.slider('Âge', 20, 100, 50)
        
        col1, col2 = st.columns(2)
        with col1:
            sex = st.selectbox('Sexe', ['Homme', 'Femme'])
        with col2:
            fbs = st.selectbox('Glycémie à jeun > 120 mg/dl', ['Non', 'Oui'])
        
        cp = st.selectbox('Type de douleur thoracique', 
                         ['Angine typique', 'Angine atypique', 'Douleur non angineuse', 'Asymptomatique'])
        
        col3, col4 = st.columns(2)
        with col3:
            trestbps = st.slider('Pression artérielle (mm Hg)', 90, 200, 120)
        with col4:
            chol = st.slider('Cholestérol (mg/dl)', 100, 600, 250)
        
        restecg = st.selectbox('Résultats ECG au repos', 
                              ['Normal', 'Anomalie de l\'onde ST-T', 'Hypertrophie probable/certaine'])
        
        col5, col6 = st.columns(2)
        with col5:
            thalach = st.slider('Fréq. cardiaque max', 70, 220, 150)
        with col6:
            exang = st.selectbox('Angine due à l\'exercice', ['Non', 'Oui'])
        
        oldpeak = st.slider('Dépression ST (exercice)', 0.0, 6.0, 1.0, 0.1)
        
        slope = st.selectbox('Pente du segment ST', 
                            ['Montante', 'Plate', 'Descendante'])
        
        col7, col8 = st.columns(2)
        with col7:
            ca = st.slider('Nb vaisseaux colorés', 0, 4, 0)
        with col8:
            thal = st.selectbox('Thalassémie', ['Normal', 'Défaut fixe', 'Défaut réversible'])
        
        # Conversion des variables catégorielles
        sex_dict = {'Homme': 1, 'Femme': 0}
        cp_dict = {'Angine typique': 0, 'Angine atypique': 1, 'Douleur non angineuse': 2, 'Asymptomatique': 3}
        fbs_dict = {'Non': 0, 'Oui': 1}
        restecg_dict = {'Normal': 0, 'Anomalie de l\'onde ST-T': 1, 'Hypertrophie probable/certaine': 2}
        exang_dict = {'Non': 0, 'Oui': 1}
        slope_dict = {'Montante': 0, 'Plate': 1, 'Descendante': 2}
        thal_dict = {'Normal': 1, 'Défaut fixe': 2, 'Défaut réversible': 3}
        
        data = {
            'age': age,
            'sex': sex_dict[sex],
            'cp': cp_dict[cp],
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs_dict[fbs],
            'restecg': restecg_dict[restecg],
            'thalach': thalach,
            'exang': exang_dict[exang],
            'oldpeak': oldpeak,
            'slope': slope_dict[slope],
            'ca': ca,
            'thal': thal_dict[thal]
        }
        
        return pd.DataFrame(data, index=[0])

# Contenu de l'onglet Prédiction
with tab1:
    if model_artifacts:
        # Collecte des entrées utilisateur
        df_input = user_input_features()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader('Données du patient')
            
            # Conversion pour affichage plus lisible
            display_df = df_input.copy()
            if 'sex' in display_df.columns:
                display_df['sex'] = display_df['sex'].map({0: 'Femme', 1: 'Homme'})
            if 'cp' in display_df.columns:
                display_df['cp'] = display_df['cp'].map({0: 'Angine typique', 1: 'Angine atypique', 
                                                       2: 'Douleur non angineuse', 3: 'Asymptomatique'})
            if 'fbs' in display_df.columns:
                display_df['fbs'] = display_df['fbs'].map({0: 'Non', 1: 'Oui'})
            if 'restecg' in display_df.columns:
                display_df['restecg'] = display_df['restecg'].map({0: 'Normal', 1: 'Anomalie ST-T', 
                                                                 2: 'Hypertrophie'})
            if 'exang' in display_df.columns:
                display_df['exang'] = display_df['exang'].map({0: 'Non', 1: 'Oui'})
            if 'slope' in display_df.columns:
                display_df['slope'] = display_df['slope'].map({0: 'Montante', 1: 'Plate', 2: 'Descendante'})
            if 'thal' in display_df.columns:
                display_df['thal'] = display_df['thal'].map({1: 'Normal', 2: 'Défaut fixe', 3: 'Défaut réversible'})
                
            # Renommer les colonnes pour l'affichage
            display_df.columns = [
                'Âge', 'Sexe', 'Type douleur', 'Pression artérielle', 'Cholestérol',
                'Glycémie élevée', 'ECG repos', 'Fréq. cardiaque max', 'Angine exercice',
                'Dépression ST', 'Pente ST', 'Vaisseaux colorés', 'Thalassémie'
            ]
            
            st.dataframe(display_df, use_container_width=True)
        
        if st.button('Analyser et prédire', use_container_width=True, type="primary"):
            with st.spinner('Analyse en cours...'):
                try:
                    # 1. S'assurer que le DataFrame a les colonnes originales dans le bon ordre
                    df_input_ordered = df_input[original_feature_names]

                    # 2. Normalisation (si le scaler existe)
                    if scaler:
                        input_processed = scaler.transform(df_input_ordered)
                    else:
                        input_processed = df_input_ordered.values
                        st.warning("Avertissement: Scaler non trouvé. La prédiction est effectuée sans normalisation.")
                    
                    # 3. Sélection de caractéristiques (si le sélecteur existe)
                    if selector:
                        input_processed = selector.transform(input_processed)

                    # 4. Prédiction
                    prediction = model.predict(input_processed)
                    
                    # Vérifier si le modèle peut prédire des probabilités
                    if hasattr(model, 'predict_proba'):
                        prediction_proba = model.predict_proba(input_processed)
                        has_proba = True
                    else:
                        has_proba = False

                    # === Affichage des résultats (votre code original, qui est excellent) ===
                    
                    st.header('Résultat de la prédiction')
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        if prediction[0] == 1:
                            st.error('⚠️ Risque de maladie cardiaque détecté')
                            if has_proba:
                                st.warning(f"Probabilité de risque: {prediction_proba[0][1]:.2%}")
                        else:
                            st.success('✅ Faible risque de maladie cardiaque détecté')
                            if has_proba:
                                st.info(f"Probabilité de ne pas avoir la maladie: {prediction_proba[0][0]:.2%}")
                    
                    with col2:
                        if has_proba:
                            # Visualisation de la probabilité en camembert
                            fig, ax = plt.subplots(figsize=(4, 3))
                            labels = ['Sain', 'Risque']
                            sizes = [prediction_proba[0][0], prediction_proba[0][1]]
                            colors = ['#2ecc71', '#e74c3c']
                            explode = (0, 0.1) if prediction[0] == 1 else (0, 0)
                            
                            ax.pie(sizes, explode=explode, labels=labels, colors=colors, 
                                autopct='%1.1f%%', shadow=True, startangle=90)
                            ax.axis('equal')
                            st.pyplot(fig)
                    
                    # Conseils basés sur les résultats
                    st.subheader('Recommandations')
                    if prediction[0] == 1:
                        st.warning("""
                        - **Consultez un cardiologue** pour une évaluation complète.
                        - Suivez un régime alimentaire adapté (pauvre en sel et graisses saturées).
                        - Pratiquez une activité physique régulière et modérée, selon avis médical.
                        - Surveillez attentivement votre tension artérielle et votre cholestérol.
                        """)
                    else:
                        st.info("""
                        - Continuez à maintenir de bonnes habitudes alimentaires et un mode de vie actif.
                        - Effectuez des contrôles médicaux périodiques pour un suivi préventif.
                        """)
                    
                    # Visualisation des facteurs de risque
                    st.subheader('Analyse des facteurs de risque du patient')
                    
                    risk_factors = pd.DataFrame({
                        'Facteur': ['Âge', 'Cholestérol', 'Fréq. cardiaque max', 'Angine exercice', 'Dépression ST', 'Nb vaisseaux colorés'],
                        'Valeur': [df_input['age'][0], df_input['chol'][0], df_input['thalach'][0], 
                                'Oui' if df_input['exang'][0] == 1 else 'Non', 
                                df_input['oldpeak'][0], df_input['ca'][0]],
                        'Niveau de Risque': [
                            'Élevé' if df_input['age'][0] > 60 else 'Modéré' if df_input['age'][0] > 45 else 'Faible',
                            'Élevé' if df_input['chol'][0] > 240 else 'Modéré' if df_input['chol'][0] > 200 else 'Faible',
                            'Élevé' if df_input['thalach'][0] < 120 else 'Modéré' if df_input['thalach'][0] < 150 else 'Faible',
                            'Élevé' if df_input['exang'][0] == 1 else 'Faible',
                            'Élevé' if df_input['oldpeak'][0] > 2 else 'Modéré' if df_input['oldpeak'][0] > 1 else 'Faible',
                            'Élevé' if df_input['ca'][0] > 1 else 'Modéré' if df_input['ca'][0] == 1 else 'Faible'
                        ]
                    })
                    
                    def color_risk(val):
                        if val == 'Élevé': return 'background-color: #ffcccc'
                        elif val == 'Modéré': return 'background-color: #ffffcc'
                        else: return 'background-color: #ccffcc'
                    
                    st.dataframe(risk_factors.style.applymap(color_risk, subset=['Niveau de Risque']), use_container_width=True)

                except Exception as e:
                    st.error(f"Erreur lors de la prédiction: {e}")
                    st.info("Veuillez vérifier que les données d'entrée sont correctes et que le modèle chargé est compatible.")



        # Prédiction
        # NOUVEAU CODE CORRIGÉ COMPLET POUR LE BOUTON "Analyser et prédire"

    # else:
    #     st.warning("⚠️ Veuillez d'abord charger un modèle valide.")

# Contenu de l'onglet À propos du modèle
with tab2:
    st.header("À propos du modèle de prédiction")
    
    st.markdown("""
    ### Description du projet
    
    Ce projet utilise des algorithmes de machine learning pour prédire le risque de maladie cardiaque
    en se basant sur diverses caractéristiques médicales des patients. Le modèle a été entraîné sur le 
    dataset Cleveland et validé sur le dataset Statlog, tous deux référencés dans la littérature médicale.
    
    ### Caractéristiques utilisées
    
    Le modèle prend en compte les caractéristiques suivantes:
    
    | Caractéristique | Description |
    | --- | --- |
    | age | Âge du patient en années |
    | sex | Sexe (1 = homme, 0 = femme) |
    | cp | Type de douleur thoracique (0-3) |
    | trestbps | Pression artérielle au repos (mm Hg) |
    | chol | Cholestérol sérique (mg/dl) |
    | fbs | Glycémie à jeun > 120 mg/dl (1 = vrai, 0 = faux) |
    | restecg | Résultats électrocardiographiques au repos (0-2) |
    | thalach | Fréquence cardiaque maximale atteinte |
    | exang | Angine induite par l'exercice (1 = oui, 0 = non) |
    | oldpeak | Dépression ST induite par l'exercice |
    | slope | Pente du segment ST à l'exercice (0-2) |
    | ca | Nombre de vaisseaux majeurs colorés par fluoroscopie (0-4) |
    | thal | Thalassémie (1 = normal, 2 = défaut fixe, 3 = défaut réversible) |
    
    ### Algorithmes utilisés
    
    Plusieurs algorithmes ont été évalués pour cette tâche de prédiction:
    
    - Régression logistique
    - Arbre de décision
    - K plus proches voisins (KNN)
    - Forêt aléatoire (Random Forest)
    - AdaBoost
    - Machines à vecteurs de support (SVM)
    
    Le modèle final a été sélectionné en fonction de ses performances sur les métriques suivantes:
    accuracy, precision, recall, F1-score et AUC.
    """)
    
    # Affichage des caractéristiques importantes si disponibles
    # NOUVEAU CODE CORRIGÉ
    if model_artifacts and hasattr(model, 'feature_importances_'):
        st.subheader("Importance des caractéristiques")
        
        # Le code s'adapte maintenant dynamiquement aux noms de features chargés
        importance_df = pd.DataFrame({
            'Caractéristique': feature_names, # Utilise les noms de features chargés
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Le reste de votre code pour afficher le graphique et le dataframe est bon
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Caractéristique', data=importance_df, palette='viridis')
        plt.title('Importance des caractéristiques dans le modèle', fontsize=14)
        plt.xlabel('Importance relative')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown(get_figure_download_link(fig, "feature_importance.png", "📥 Télécharger le graphique"), unsafe_allow_html=True)
        st.dataframe(importance_df, use_container_width=True)
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Afficher un graphique
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Caractéristique', data=importance_df, palette='viridis')
        plt.title('Importance des caractéristiques dans le modèle', fontsize=14)
        plt.xlabel('Importance relative')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Lien de téléchargement
        st.markdown(get_figure_download_link(fig, "feature_importance.png", "📥 Télécharger le graphique"), unsafe_allow_html=True)
        
        # Afficher les valeurs dans un tableau
        st.dataframe(importance_df, use_container_width=True)

# Contenu de l'onglet Statistiques
with tab3:
    st.header("Statistiques et informations sur les maladies cardiaques")
    
    # Données statiques sur les maladies cardiaques
    st.subheader("Facteurs de risque des maladies cardiaques")
    
    risk_cols = st.columns(3)
    
    with risk_cols[0]:
        st.markdown("""
        **Facteurs non modifiables:**
        - Âge avancé
        - Sexe masculin (risque plus élevé)
        - Antécédents familiaux
        - Origine ethnique
        """)
    
    with risk_cols[1]:
        st.markdown("""
        **Facteurs liés au mode de vie:**
        - Tabagisme
        - Alimentation riche en graisses
        - Sédentarité
        - Consommation excessive d'alcool
        - Stress chronique
        """)
    
    with risk_cols[2]:
        st.markdown("""
        **Conditions médicales:**
        - Hypertension artérielle
        - Cholestérol élevé
        - Diabète
        - Obésité
        - Syndrome métabolique
        """)
    
    # Graphique mondial des maladies cardiaques
    st.subheader("Statistiques mondiales")
    
    world_stats = {
        "Région": ["Amérique du Nord", "Europe", "Asie du Sud", "Asie de l'Est", "Afrique", "Amérique Latine", "Océanie"],
        "Prévalence (%)": [6.1, 5.4, 4.8, 3.9, 3.7, 4.2, 5.1]
    }
    
    world_df = pd.DataFrame(world_stats)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Région", y="Prévalence (%)", data=world_df, palette="coolwarm")
    plt.title("Prévalence des maladies cardiaques par région du monde", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Section prévention
    st.subheader("Prévention des maladies cardiaques")
    
    prevention_cols = st.columns(2)
    
    with prevention_cols[0]:
        st.markdown("""
        ### Recommandations alimentaires
        
        - Réduire la consommation de graisses saturées et de sel
        - Augmenter la consommation de fruits et légumes
        - Privilégier les graisses insaturées (huile d'olive, noix)
        - Limiter la viande rouge
        - Éviter les aliments transformés
        - Surveiller sa consommation d'alcool
        """)
    
    with prevention_cols[1]:
        st.markdown("""
        ### Activité physique recommandée
        
        - 150 minutes d'activité modérée par semaine
        - Ou 75 minutes d'activité intense par semaine
        - Intégrer des exercices de renforcement musculaire
        - Pratiquer au moins 10 minutes d'activité continue
        - Réduire les périodes d'inactivité prolongée
        - Consulter un médecin avant de commencer un programme intensif
        """)
    
    # Ajouter une note de bas de page
    st.markdown("---")
    st.caption("""
    **Note importante:** Cette application est fournie à des fins éducatives uniquement et ne remplace pas l'avis médical professionnel. 
    Consultez toujours un médecin pour des questions relatives à votre santé.
    """)

# Afficher les informations sur l'application en bas de page
with st.sidebar.expander("À propos de cette application"):
    st.markdown("""
    **Développée par:** Team ABBNWY

    **Version:** 1.0

    **Dernière mise à jour:** Mai 2025
    
    Cette application est basée sur des modèles de machine learning
    entraînés sur des données médicales publiques.
    """)
