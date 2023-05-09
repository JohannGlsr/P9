import pandas as pd
from surprise import Reader, Dataset, KNNBasic
from surprise.model_selection import train_test_split
import pickle

def recommander_articles(user, data_users, model):
    
    # Convertir les données utilisateurs au format de Surprise
    reader = Reader(rating_scale=(0, 1))
    user_id = user
    articles_lus = data_users.loc[data_users['user_id'] == user_id, 'click_article_id'].tolist()
    articles_non_lus = list(set(data_users['click_article_id']) - set(articles_lus))

    user_items = []
    for article_id in articles_lus:
        user_items.append((user_id, article_id, 1))

    for article_id in articles_non_lus:
        user_items.append((user_id, article_id, 0))

    df_user_items = pd.DataFrame(user_items, columns=['user_id', 'click_article_id', 'session_size'])
    data_user_items = Dataset.load_from_df(df_user_items[['user_id', 'click_article_id', 'session_size']].astype(int), reader)
    testset = data_user_items.build_full_trainset().build_testset()
    predictions = model.test(testset)

    recommended_articles = []
    for user_id, article_id, _, predicted_rating, _ in predictions:
        if user_id == user and predicted_rating > 0:
            recommended_articles.append(article_id)

    return recommended_articles[0:5]

data_users = pd.read_csv('clicks_sample.csv')

reader = Reader(rating_scale=(0, 1))
data = Dataset.load_from_df(data_users[['user_id', 'click_article_id', 'session_size']].astype(int), reader)

trainset, testset = train_test_split(data, test_size=.25)

sim_options = {'name': 'cosine', 'user_based': True}
model = KNNBasic(sim_options=sim_options)
model.fit(trainset)

with open('CollaborativeFilteringRecommenderModel.pkl', 'wb') as file:
    pickle.dump(model, file)