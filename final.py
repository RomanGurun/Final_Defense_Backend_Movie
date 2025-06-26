from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pandas as pd
import requests
import random
import bs4
import re
from tmdbv3api import TMDb, Movie
import pickle as pkl
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
import numpy as np
from urllib.parse import unquote

app = Flask(__name__)
CORS(app)

# TMDb Setup
tmdb = TMDb()
tmdb.api_key = '2c5341f7625493017933e27e81b1425e'
tmdb_movie = Movie()

# Load data and models
df2 = pd.read_csv("tmdb_5000_credits.csv")
knn1 = pd.read_csv("tmdb_5000_movies.csv")
vectorizer = pkl.load(open('vectorizerer.pkl', 'rb'))
clt = pkl.load(open('nlp_model.pkl', 'rb'))

# URLs for movie discovery
url_list = [
    "https://api.themoviedb.org/3/discover/movie?api_key=2c5341f7625493017933e27e81b1425e&primary_release_year=2015&adult=false",
    "http://api.themoviedb.org/3/discover/movie?api_key=2c5341f7625493017933e27e81b1425e&primary_release_year=2014&adult=false",
    "https://api.themoviedb.org/3/movie/popular?api_key=2c5341f7625493017933e27e81b1425e&language=en-US&page=1&adult=false",
    # ... add other URLs as needed ...
]

# Helper functions

def get_news():
    response = requests.get("https://www.imdb.com/news/top/?ref_=hm_nw_sm")
    soup = bs4.BeautifulSoup(response.text, 'html.parser')
    articles = []
    texts = [re.sub('[\n()]', "", d.text) for d in soup.find_all('div', class_='news-article__content')]
    images = [img['src'] for img in soup.find_all("img", class_="news-article__image")]
    for i in range(len(texts)):
        articles.append([images[i], texts[i].strip()])
    return articles

def get_director(movie_title):
    search_results = tmdb_movie.search(movie_title)
    if not search_results:
        return []
    movie_id = search_results[0].id
    response = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={tmdb.api_key}")
    data_json = response.json()
    crew = data_json.get('crew', [])
    for member in crew:
        if member.get('job') == 'Director':
            return [member.get('name')]
    return []

def get_swipe():
    data = []
    base_url = random.choice(url_list)
    for page in range(1, 6):
        response = requests.get(f"{base_url}&page={page}")
        results = response.json().get("results", [])
        data.extend(results)
    return data

def get_reviews(movie_title):
    search_results = tmdb_movie.search(movie_title)
    if not search_results:
        return []
    movie_id = search_results[0].id
    response = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}/reviews?api_key={tmdb.api_key}&language=en-US&page=1")
    data_json = response.json()
    return data_json.get('results', [])

def get_rating_reviews(movie_title):
    reviews = get_reviews(movie_title)
    movie_review = []
    for review in reviews:
        pred = clt.predict(vectorizer.transform([review['content']]))
        rating = "Good" if pred[0] == 'positive' else "Bad"
        movie_review.append({"review": review['content'], "rating": rating})
    return movie_review

def get_movie_data(movie_title):
    search_results = tmdb_movie.search(movie_title)
    if not search_results:
        return []
    movie_id = search_results[0].id
    details = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={tmdb.api_key}").json()
    credits = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={tmdb.api_key}").json()
    keywords = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}/keywords?api_key={tmdb.api_key}").json()
    return [details, credits, keywords]

def combine_movie_data(movie_data):
    cast = [c['name'] for c in movie_data[1].get('cast', [])]
    director = [c['name'] for c in movie_data[1].get('crew', []) if c['job'] == 'Director']
    genres = [g['name'] for g in movie_data[0].get('genres', [])]
    keywords = [k['name'] for k in movie_data[2].get('keywords', [])]
    overview = movie_data[0].get('overview', '')
    combined_str = str(cast) + str(keywords) + str(genres) + (director[0] if director else '') + overview
    return combined_str

def get_movie_and_trailer(movie_title):
    search_results = tmdb_movie.search(movie_title)
    if not search_results:
        return []
    movie_id = search_results[0].id
    details = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={tmdb.api_key}").json()
    trailer = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={tmdb.api_key}&language=en-US").json()
    return [details, trailer]

# Flask routes

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/getname', methods=["GET"])
def get_names():
    titles = df2["title_x"].tolist()
    return jsonify(titles)

@app.route('/getmovie/<movie_name>', methods=["GET"])
def get_movie(movie_name):
    data = get_movie_and_trailer(movie_name)
    return jsonify(data)

@app.route('/getreview/<movie_name>', methods=["GET"])
def get_reviews_route(movie_name):
    data = get_rating_reviews(movie_name)
    return jsonify(data)

@app.route('/getdirector/<movie_name>', methods=["GET"])
def get_director_route(movie_name):
    data = get_director(movie_name)
    return jsonify(data)

@app.route('/getswipe', methods=["GET"])
def get_swipe_route():
    data = get_swipe()
    return jsonify(data)

@app.route('/getnews', methods=["GET"])
def get_news_route():
    data = get_news()
    return jsonify(data)

# Recommendations (content-based, collaborative, hybrid)

def get_recommendations(title, user_id):
    movies_data = pd.read_csv('Main_data.csv')
    ratings_data = pd.read_csv('movie_rating.csv')

    def content_based_recommendations(title, movies_data, top_n=6):
        movies_data['comb'] = movies_data['title_x'].fillna('') + movies_data['genres'].fillna('')
        if not movies_data['title_x'].str.contains(title).any():
            movies_data = movies_data.append({'title_x': title, 'genres': ''}, ignore_index=True)
        tfidf = TfidfVectorizer(stop_words='english')
        count_matrix = tfidf.fit_transform(movies_data['comb'])
        idx = movies_data[movies_data['title_x'] == title].index[0]
        cosine_sim = cosine_similarity(count_matrix, count_matrix)
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
        movie_indices = [i[0] for i in sim_scores]
        return movies_data['title_x'].iloc[movie_indices]

    def collaborative_filtering_recommendations(user_id, top_n=10):
        merged_data = pd.merge(movies_data, ratings_data, left_on='id', right_on='movieId')
        user_item_matrix = pd.pivot_table(data=merged_data, values='rating', index='userId', columns='movieId', fill_value=0)
        if user_id not in user_item_matrix.index:
            print("User ID doesn't exist.")
            return pd.Series()
        item_similarity = 1 - pairwise_distances(user_item_matrix.T, metric='cosine')
        min_rating = user_item_matrix.min().min()
        max_rating = user_item_matrix.max().max()
        normalized_ratings = (user_item_matrix - min_rating) / (max_rating - min_rating)
        user_ratings = normalized_ratings.loc[user_id].values.reshape(1, -1)
        predicted_ratings = np.dot(user_ratings, item_similarity) / np.sum(item_similarity)
        predicted_ratings = predicted_ratings * (max_rating - min_rating) + min_rating
        top_movies_indices = np.argsort(-predicted_ratings[0])[:top_n]
        top_movies = movies_data[movies_data['id'].isin(top_movies_indices)]['title_x']
        return top_movies

    def hybrid_recommendations(user_id, movie_title):
        content_recs = content_based_recommendations(movie_title, movies_data)
        collaborative_recs = collaborative_filtering_recommendations(user_id)
        hybrid_recs = pd.concat([content_recs, collaborative_recs]).drop_duplicates().reset_index(drop=True)
        return hybrid_recs

    return hybrid_recommendations(user_id, title)

@app.route('/send/<movie_name>/<string:userId>', methods=["GET"])
def send_recommendations(movie_name, userId):
    print(f"Request received for movie: {movie_name}, userId: {userId}")
    recs = get_recommendations(movie_name, userId)
    if recs.empty:
        return jsonify({"message": "movie not found in database"})
    results = []
    for movie in recs:
        res = get_movie_and_trailer(movie)
        if res:
            results.append(res[0])
    return jsonify(results)

@app.route('/rate/<movieId>/<float:rate>/<string:userId>', methods=["GET"])
def rate_movie(movieId, rate, userId):
    import csv
    data = {'userId': userId, 'movieId': movieId, 'rating': rate}
    filename = 'movie_rating.csv'
    with open(filename, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['userId', 'movieId', 'rating'])
        writer.writerow(data)
    return jsonify(data)

@app.route('/score/<path:title1>/<path:title2>/', methods=["GET"])
def find_score(title1, title2):
    title1 = unquote(title1)
    title2 = unquote(title2)
    movies_data = pd.read_csv('Main_data.csv')
    movies_data['comb'] = movies_data['title_x'].fillna('') + movies_data['genres'].fillna('')

    try:
        index1 = movies_data[movies_data['title_x'] == title1].index[0]
        index2 = movies_data[movies_data['title_x'] == title2].index[0]
    except IndexError:
        return jsonify({"error": "One or both movie titles not found"}), 404

    # cosine similarity
    count_vectorizer = CountVectorizer()
    count_matrix = count_vectorizer.fit_transform(movies_data['comb'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    similarity = cosine_sim[index1, index2]

    # euclidean and manhattan distances
    tfidf = TfidfVectorizer(stop_words='english')
    feature_matrix = tfidf.fit_transform(movies_data)