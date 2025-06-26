# app.py (refactored Flask backend)

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import requests
import json
import random
import csv
import re
import pickle as pkl
import bs4
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.metrics import pairwise_distances
from tmdbv3api import TMDb, Movie
from urllib.parse import unquote
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Flask app setup
app = Flask(__name__)
CORS(app)

# TMDb setup
tmdb = TMDb()
tmdb.api_key = '2c5341f7625493017933e27e81b1425e'
tmdb_movie = Movie()

# Load data and models
df2 = pd.read_csv("tmdb_5000_credits.csv")
knn1 = pd.read_csv("tmdb_5000_movies.csv")
vectorizer = pkl.load(open('vectorizerer.pkl', 'rb'))
clt = pkl.load(open('nlp_model.pkl', 'rb'))

# Utility URLs
urls = [
    f"https://api.themoviedb.org/3/discover/movie?api_key={tmdb.api_key}&primary_release_year={year}&adult=false"
    for year in range(2014, 2021)
] + [
    f"https://api.themoviedb.org/3/movie/popular?api_key={tmdb.api_key}&language=en-US&page={page}&adult=false"
    for page in range(1, 4)
] + [
    f"https://api.themoviedb.org/3/discover/movie?api_key={tmdb.api_key}&with_genres={genre}"
    for genre in [18, 27, 16]
]

# Helper functions (omitting unchanged implementations for brevity)
def get_news():
    ...

def get_data(movie_name):
    ...

def get_data2(movie_name):
    ...

def get_comb(movie_data):
    ...

def get_swipe():
    ...

def get_rating(title):
    ...

def get_director(title):
    ...

# Flask routes
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/getname', methods=['GET'])
def get_names():
    return jsonify(df2["title_x"].tolist())

@app.route('/getmovie/<movie_name>', methods=['GET'])
def get_movie(movie_name):
    return jsonify(get_data2(movie_name))

@app.route('/getreview/<movie_name>', methods=['GET'])
def get_reviews(movie_name):
    return jsonify(get_rating(movie_name))

@app.route('/getdirector/<movie_name>', methods=['GET'])
def get_director_name(movie_name):
    return jsonify(get_director(movie_name))

@app.route('/getswipe', methods=['GET'])
def get_swipes():
    return jsonify(get_swipe())

@app.route('/getnews', methods=['GET'])
def get_news_data():
    return jsonify(get_news())

@app.route('/send/<movie_name>/<string:userId>', methods=['GET'])
def hybrid_recommend(movie_name, userId):
    movies_data = pd.read_csv('Main_data.csv')
    ratings_data = pd.read_csv('movie_rating.csv')

    def content_based(title):
        movies_data['comb'] = movies_data['title_x'] + movies_data['genres']
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(movies_data['comb'].fillna(''))
        idx = movies_data[movies_data['title_x'] == title].index[0]
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:7]
        return movies_data.iloc[[i[0] for i in sim_scores]]['title_x']

    def collaborative(user_id):
        merged = pd.merge(movies_data, ratings_data, left_on='id', right_on='movieId')
        user_item = pd.pivot_table(merged, values='rating', index='userId', columns='movieId', fill_value=0)
        if user_id not in user_item.index:
            return []
        item_sim = pairwise_distances(user_item.T, metric='cosine')
        user_ratings = user_item.loc[user_id].values.reshape(1, -1)
        pred = np.dot(user_ratings, item_sim) / np.sum(item_sim)
        top_ids = np.argsort(-pred[0])[:10]
        return movies_data[movies_data['id'].isin(top_ids)]['title_x']

    hybrid_list = pd.concat([content_based(movie_name), collaborative(userId)])
    result = []
    for title in hybrid_list.unique():
        try:
            result.append(get_data2(title)[0])
        except Exception:
            continue
    return jsonify(result)

@app.route('/rate/<movieId>/<float:rate>/<string:userId>', methods=['GET'])
def rate_movie(movieId, rate, userId):
    with open('movie_rating.csv', 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=['userId', 'movieId', 'rating']).writerow({'userId': userId, 'movieId': movieId, 'rating': rate})
    return jsonify({'userId': userId, 'movieId': movieId, 'rating': rate})

@app.route('/review/<movieId>/<string:review>/<string:userId>', methods=['GET'])
def review_movie(movieId, review, userId):
    with open('IMDB Dataset.csv', 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=['userId', 'movieId', 'review']).writerow({'userId': userId, 'movieId': movieId, 'review': review})
    return jsonify({'userId': userId, 'movieId': movieId, 'review': review})

@app.route('/store/<movieId>/<string:movie1>/<string:userId>', methods=['GET'])
def store_movie(movieId, movie1, userId):
    with open('Main_data.csv', 'a', newline='') as f:
        csv.writer(f).writerow([userId, movieId, movie1])
    return jsonify({'userId': userId, 'movieId': movieId, 'movie1': movie1})

@app.route('/score/<path:title1>/<path:title2>/', methods=['GET'])
def score(title1, title2):
    title1, title2 = unquote(title1), unquote(title2)
    df = pd.read_csv('Main_data.csv')
    df['comb'] = df['title_x'] + df['genres']
    idx1, idx2 = df[df['title_x'] == title1].index[0], df[df['title_x'] == title2].index[0]
    count_matrix = CountVectorizer().fit_transform(df['comb'])
    cosine_sim = cosine_similarity(count_matrix)
    tfidf = TfidfVectorizer(stop_words='english')
    features = tfidf.fit_transform(df['comb']).toarray()
    sim = cosine_sim[idx1, idx2]
    euc = euclidean_distances(features[idx1].reshape(1, -1), features[idx2].reshape(1, -1))[0][0]
    manh = manhattan_distances(features[idx1].reshape(1, -1), features[idx2].reshape(1, -1))[0][0]
    return jsonify({'cosineSimilarity': sim, 'euclideanDistance': euc, 'manhattanDistance': manh})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)




