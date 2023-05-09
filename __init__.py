import logging
import azure.functions as func
import pandas as pd
import pickle
from surprise import Reader, Dataset, KNNBasic
from io import BytesIO
from azure.storage.blob import BlobServiceClient

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    user_id = req.params.get('user_id')
    if not user_id:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            user_id = req_body.get('user_id')

    if user_id:
        container_name = "conteneurcsv"
        blob_name = "clicks_sample.csv"
        connection_string = "DefaultEndpointsProtocol=https;AccountName=stockagecsvprojet;AccountKey=JXP1ZM8wPf74fdTE7bQKUf4RFFb5hUZMM09a3Mw8gpMEDrXG544q0daDdBInZbmLdaEbk5U/Cyv3+ASto1ALoA==;EndpointSuffix=core.windows.net"

        # Récupérer les données utilisateur depuis un fichier CSV stocké dans un conteneur Blob
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        stream = blob_client.download_blob()
        data_users = pd.read_csv(BytesIO(stream.readall()))

        # Charger le modèle pré-entraîné
        blob_name = "CollaborativeFilteringRecommenderModel.pkl"
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        stream = blob_client.download_blob()
        model = pickle.load(BytesIO(stream.readall()))

        # Recommander des articles pour l'utilisateur donné
        articles_lus = data_users.loc[data_users['user_id'] == int(user_id), 'click_article_id'].tolist()
        articles_non_lus = list(set(data_users['click_article_id']) - set(articles_lus))

        user_items = []
        for article_id in articles_lus:
            user_items.append((int(user_id), article_id, 1))

        for article_id in articles_non_lus:
            user_items.append((int(user_id), article_id, 0))

        df_user_items = pd.DataFrame(user_items, columns=['user_id', 'click_article_id', 'session_size'])
        reader = Reader(rating_scale=(0, 1))
        data_user_items = Dataset.load_from_df(df_user_items[['user_id', 'click_article_id', 'session_size']].astype(int), reader)
        testset = data_user_items.build_full_trainset().build_testset()
        predictions = model.test(testset)

        recommended_articles = []
        for user_id, article_id, _, predicted_rating, _ in predictions:
            if user_id == int(user_id) and predicted_rating > 0:
                recommended_articles.append(article_id)

        return func.HttpResponse(f"Recommandations pour l'utilisateur {user_id}: {recommended_articles[0:5]}")
    else:
        return func.HttpResponse(
             "Please provide a user_id parameter in the query string or in the request body",
             status_code=400
        )
