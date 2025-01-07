import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
import joblib
import time
from sklearn.preprocessing import StandardScaler


data = pd.read_csv("https://raw.github.com/YvesLubalika/Projet_Groupe1_Cours_IA_-_G-nie_Logiciel/blob/main/AirQualityUCI.csv", sep=";", decimal=",")

data.head()

data1 = data.iloc[:9357,:15]

#Remplacer les valeurs manquantes (-200) par NaN et les supprimer
data1.replace(-200, float('nan'), inplace=True)
data1.dropna(inplace=True)

X = data1[[ 'PT08.S1(CO)','NMHC(GT)','C6H6(GT)','PT08.S2(NMHC)', 'PT08.S3(NOx)','NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)','T', 'RH', 'AH',]]
y = data1[['CO(GT)', 'NOx(GT)']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliser les caractéristiques
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Créer et entraîner le modèle Random_Forest 
Random_Forest = RandomForestRegressor(n_estimators=100, random_state=42)
Random_Forest.fit(X_train, y_train)
# Faire des prédictions
y_pred = Random_Forest.predict(X_test)
r2 = r2_score(y_test, y_pred)



import streamlit as st
import numpy as np
import fitz  # PyMuPDF pour affichage des PDF
import base64
import os

# Définir la mise en page large
st.set_page_config(layout='wide')
def page1():
    # Créer trois colonnes avec des largeurs différentes
    col1, col2, col3 = st.columns([1, 2, 1])
    # Ajouter du contenu dans chaque colonne
    with col1:
        st.markdown("")
       
    with col2:
        st.markdown(
                    """
                    <div style="text-align: center;">
                    <h1>UNIVERSITE OFFICILLE DE BUKAVU</h1>
                    </div>
                    """,
                    unsafe_allow_html=True
                    )
    st.write("")    
    st.markdown(
                """
                <div style="text-align: center;">
                <h1>PROJET DE MACHINE LEARNING ET DE GENIE LOGICIEL</h1>
                </div>
                """,
                unsafe_allow_html=True
                )

    col4, col5, col6 = st.columns([1, 2, 1])
    with col5:
        st.markdown(
                    """
                    <div style="text-align: center;">
                    <h1>Ecole Des Mines</h1>
                    </div>
                    """,
                    unsafe_allow_html=True
                    )
        
    col7, col8, col9, col10 = st.columns([1, 3, 3, 2])
    with col7:
        st.write("")
        
    with col8:
        st.write("")
    with col9:
        st.subheader('')
        st.write("")
        st.subheader("Présenter par :")
                    
    with col10:
        st.subheader('')
        st.write("")
        st.write("")
        st.write("BUHENDWA  AKONKWA  Josué")
        st.write("ESPOIR  MUJANGA  Habineza")
        st.write("IRAGI  LUBALIKA  Yves")
        
    col11, col12, col13, col14 = st.columns([1, 3, 3, 2])
    with col13:
        st.write("")
        st.write("")
        st.subheader("Dispensé par : ")
                                    
    with col14:
        st.write("")
        st.write("")
        st.write("")
        st.write("AGISHA  NTWALI  Albert")
        
    col15, col16, col17 = st.columns([1, 2, 1])
    with col16:
        st.subheader('')
        st.markdown(
                    """
                    <div style="text-align: center;">
                    <h1>Année Académique: 2023-2024</h1>
                    </div>
                    """,
                    unsafe_allow_html=True
                    )


def page2():
    # Titre de l'application
    st.write ('''
    # Prédiction de la Qualité de lAir dans les Zones Minières
    ''')
    # Formulaire de saisie des données
    st.sidebar.header("Entrer les données météorologiques et dexploitation")
    def user_input():
        input_method = st.sidebar.radio("Choisissez la méthode de saisie", ("Slider", "Input manuel"))
        
        if input_method == "Slider":
            sensor1 = st.sidebar.slider('Réponse du capteur 1 (CO)', 648, 2041, 1000)
            sensor2 = st.sidebar.slider('Concentration des hydrocarbures non méthaniques (NMHC)', 8, 1189, 500)
            sensor3 = st.sidebar.slider('Concentration de benzène (C6H6)', 0, 64, 10)
            sensor4 = st.sidebar.slider('Réponse du Capteur 2 (NMHC)', 383, 2215, 1000)
            sensor5 = st.sidebar.slider('Réponse du Capteur 3 (NOx)', 323, 2684, 1000)
            sensor6 = st.sidebar.slider('Concentration de dioxyde d\'azote (NO2)', 1, 341, 100)
            sensor7 = st.sidebar.slider('Réponse du Capteur 4 (NO2)', 550, 2776, 1000)
            sensor8 = st.sidebar.slider('Réponse du Capteur 5 (O3)', 220, 2524, 1000)
            temperature = st.sidebar.slider('Température (°C)', -2, 46, 20)
            humidity = st.sidebar.slider('Humidité (%)', 9, 90, 50)
            absolute_humidity = st.sidebar.slider('Humidité Absolue (g/m³)', 0, 3, 1)
        else:
            sensor1 = st.sidebar.number_input('Réponse du capteur 1 (CO)', 648, 2041, 1000)
            sensor2 = st.sidebar.number_input('Concentration des hydrocarbures non méthaniques (NMHC)', 8, 1189, 500)
            sensor3 = st.sidebar.number_input('Concentration de benzène (C6H6)', 0, 64, 10)
            sensor4 = st.sidebar.number_input('Réponse du Capteur 2 (NMHC)', 383, 2215, 1000)
            sensor5 = st.sidebar.number_input('Réponse du Capteur 3 (NOx)', 323, 2684, 1000)
            sensor6 = st.sidebar.number_input('Concentration de dioxyde d\'azote (NO2)', 1, 341, 100)
            sensor7 = st.sidebar.number_input('Réponse du Capteur 4 (NO2)', 550, 2776, 1000)
            sensor8 = st.sidebar.number_input('Réponse du Capteur 5 (O3)', 220, 2524, 1000)
            temperature = st.sidebar.number_input('Température (°C)', -2, 46, 20)
            humidity = st.sidebar.number_input('Humidité (%)', 9, 90, 50)
            absolute_humidity = st.sidebar.number_input('Humidité Absolue (g/m³)', 0, 3, 1)
        data2={
               'PT08.S1(CO)': sensor1,
               'NMHC(GT)': sensor2,
               'C6H6(GT)': sensor3,
               'PT08.S2(NMHC)': sensor4,
               'PT08.S3(NOx)': sensor5,
               'NO2(GT)': sensor6,
               'PT08.S4(NO2)': sensor7,
               'PT08.S5(O3)': sensor8,
               'T': temperature,
               'RH': humidity,
               'AH': absolute_humidity,
              }
        parametres_air=pd.DataFrame(data2, index=[0])
        return parametres_air
    df=user_input()
    st.subheader('Les inputs')
    st.write(df)
    st.write (f'Coefficient de determination : {r2:.2f}')
    # Normaliser les nouvelles données d'entrée
    df_scaled = scaler.transform(df)
    # Prédiction
    start_time = time.time()
    prediction = Random_Forest.predict(df_scaled)
    end_time = time.time()
    execution_time = end_time - start_time

    st.subheader(f'Concentration de CO prédite : {prediction[0][0]:.2f} µg/m³')
    st.subheader(f'Concentration de NOx prédite : {prediction[0][1]:.2f} µg/m³')
    st.write(f"Temps d'exécution : {execution_time} secondes")
    # Affichage des résultats
    st.write('Cette application utilise le modèle de machine learning Random Forest pour prédire la concentration de CO et NOx en fonction des données météorologiques et des réponses des capteurs.')

def page3():
    # Créer trois colonnes avec des largeurs différentes
    col1, col2, col3 = st.columns([1, 1, 1])
        # Créer la barre latérale avec des liens vers d'autres pages
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Ouvrir :", ["Cours Bac1 Génie Minier", "Cours Bac2 Génie Minier", "Cours Bac3 Génie Minier"])
    
    # Afficher le contenu en fonction de la page sélectionnée
    if page == "Cours Bac1 Génie Minier":
        
        #Créer la fonction pour afficher le contenu d'un fichier PDF :       
        def show_pdf(file_path):
            with open(file_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
        
        #Créer la fonction pour sélectionner un sous-dossier et un fichier PDF :
        
        def file_selector(base_folder='https://github.com/YvesLubalika/Projet_Groupe1_Cours_IA_-_G-nie_Logiciel/tree/main/Cours_Bac1_G%C3%A9nie_Minier/Education%20A%20la%20Citoyennet%C3%A9%20EDC'):
            # Lister les sous-dossiers dans le dossier principal
            subfolders = [f.name for f in os.scandir(base_folder) if f.is_dir()]
            selected_subfolder = st.selectbox('Sélectionnez un sous-dossier', subfolders)
        
            # Lister les fichiers PDF dans le sous-dossier sélectionné
            folder_path = os.path.join(base_folder, selected_subfolder)
            filenames = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
            selected_filename = st.selectbox('Sélectionnez un fichier PDF', filenames)
            
            return os.path.join(folder_path, selected_filename)
         
        #Créer l'interface utilisateur :
        st.title("Premième annéé de Licence")
        
        # Afficher les fichiers PDF lorsque le chemin du dossier est fourni
        try:
            selected_file = file_selector()
            if selected_file:
                show_pdf(selected_file)
        except FileNotFoundError:
            st.error("Dossier non trouvé. Veuillez vérifier le chemin et réessayer.")

    
    elif page == "Cours Bac2 Génie Minier":
        st.title("Deuxième annéé de Licence")
         #Créer la fonction pour afficher le contenu d'un fichier PDF :       
        def show_pdf(file_path):
            with open(file_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
        
        #Créer la fonction pour sélectionner un sous-dossier et un fichier PDF :
        
        def file_selector(base_folder='https://github.com/YvesLubalika/Projet_Groupe1_Cours_IA_-_G-nie_Logiciel/tree/main/Cours_Bac2_G%C3%A9nie_Minier'):
            # Lister les sous-dossiers dans le dossier principal
            subfolders = [f.name for f in os.scandir(base_folder) if f.is_dir()]
            selected_subfolder = st.selectbox('Sélectionnez un cours', subfolders)
        
            # Lister les fichiers PDF dans le sous-dossier sélectionné
            folder_path = os.path.join(base_folder, selected_subfolder)
            filenames = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
            selected_filename = st.selectbox('Sélectionnez un fichier PDF', filenames)
            return os.path.join(folder_path, selected_filename)
         
        # Afficher les fichiers PDF lorsque le chemin du dossier est fourni
        try:
            selected_file = file_selector()
            if selected_file:
                show_pdf(selected_file)
        except FileNotFoundError:
            st.error("Dossier non trouvé. Veuillez vérifier le chemin et réessayer.")
    elif page == "Cours Bac3 Génie Minier":
         #Créer la fonction pour afficher le contenu d'un fichier PDF :       
        def show_pdf(file_path):
            with open(file_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
        
        #Créer la fonction pour sélectionner un sous-dossier et un fichier PDF :
        
        def file_selector(base_folder='https://github.com/YvesLubalika/Projet_Groupe1_Cours_IA_-_G-nie_Logiciel/tree/main/Cours_Bac3_G%C3%A9nie_Minier'):
            # Lister les sous-dossiers dans le dossier principal
            subfolders = [f.name for f in os.scandir(base_folder) if f.is_dir()]
            selected_subfolder = st.selectbox('Sélectionnez un sous-dossier', subfolders)
        
            # Lister les fichiers PDF dans le sous-dossier sélectionné
            folder_path = os.path.join(base_folder, selected_subfolder)
            filenames = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
            selected_filename = st.selectbox('Sélectionnez un fichier PDF', filenames)
            
            return os.path.join(folder_path, selected_filename)
         
        #Créer l'interface utilisateur :
        st.title("Dernière année")
        
        # Afficher les fichiers PDF lorsque le chemin du dossier est fourni
        try:
            selected_file = file_selector()
            if selected_file:
                show_pdf(selected_file)
        except FileNotFoundError:
            st.error("Dossier non trouvé. Veuillez vérifier le chemin et réessayer.")
        st.title("Bienvenue sur la Page 3")
        st.write("Contenu de la Page 3")
   
# Définir les pages disponibles
pages = {
    "Présentation_groupe": page1,
    "prédiction CO et NOx": page2,
    "Cours Licence Genie Minier UOB": page3
    }

# Utiliser st.navigation pour créer le menu de navigation
selected_page = st.sidebar.selectbox("Sélectionnez une page", pages.keys())

# Exécuter la page sélectionnée
pages[selected_page]()
