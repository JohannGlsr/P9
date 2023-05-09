import requests
import streamlit as st

# Définir la couleur de fond
st.markdown(
    """
    <style>
    body {
        background-color: #f0f5f5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# URL de votre fonction Azure
azure_function_url = "https://functionappforserverlessapp.azurewebsites.net/api/httptriggernofonction?user_id="

# Ajouter un titre à l'application
st.title("Système de recommandation d'articles")

# Créer une zone de saisie pour l'ID utilisateur
user_id = st.text_input("Entrez votre ID utilisateur")

# Ajouter un bouton pour envoyer la requête
if st.button("Recommander"):
    # Ajouter l'ID utilisateur à la requête
    params = {"user_id": user_id}

    # Envoyer la requête à la fonction Azure
    response = requests.get(azure_function_url, params=params)

    # Afficher la réponse de la fonction Azure dans l'interface utilisateur
    if response.status_code == 200:
        recommended_articles = response.text
        st.success(f"{recommended_articles}")
    else:
        st.error("Erreur lors de la récupération des recommandations.")
