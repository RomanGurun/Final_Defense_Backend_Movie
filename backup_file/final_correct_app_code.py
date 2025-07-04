

# final mac localhost runserver probelm solver 
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pandas as pd
import requests
import random
import re
import bs4
import numpy as np
import pickle as pkl
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.metrics import pairwise_distances
from tmdbv3api import TMDb, Movie
from urllib.parse import unquote
import csv
import os
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

tmdb = TMDb()
tmdb.api_key = '2c5341f7625493017933e27e81b1425e'
tmdb_movie = Movie()

# Load CSVs once at start
df2 = pd.read_csv("tmdb_5000_credits.csv")
knn1 = pd.read_csv("tmdb_5000_movies.csv")

# Load NLP vectorizer and model
with open('vectorizerer.pkl', 'rb') as f:
    vectorizer = pkl.load(f)

with open('nlp_model.pkl', 'rb') as f:
    clt = pkl.load(f)

url = [
    "https://api.themoviedb.org/3/discover/movie?api_key=2c5341f7625493017933e27e81b1425e&primary_release_year=2015&adult=false",
    "http://api.themoviedb.org/3/discover/movie?api_key=2c5341f7625493017933e27e81b1425e&primary_release_year=2014&adult=false",
    "https://api.themoviedb.org/3/movie/popular?api_key=2c5341f7625493017933e27e81b1425e&language=en-US&page=1&adult=false",
]

def get_news():
    response = requests.get("https://www.imdb.com/news/top/?ref_=hm_nw_sm")
    soup = bs4.BeautifulSoup(response.text, 'html.parser')
    data = [re.sub('[\n()]', "", d.text) for d in soup.find_all('div', class_='news-article__content')]
    image = [m['src'] for m in soup.find_all("img", class_="news-article__image")]
    t_data = []
    for i in range(len(data)):
        t_data.append([image[i], data[i][1:-1]])
    return t_data

def getdirector(x):
    result = tmdb_movie.search(x)
    movie_id = result[0].id
    response = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={tmdb.api_key}")
    data_json = response.json()
    director = [c['name'] for c in data_json['crew'] if c['job'] == 'Director']
    return director[:1]

def get_swipe():
    data = []
    val = random.choice(url)
    for i in range(5):
        response = requests.get(val + "&page=" + str(i + 1))
        data_json = response.json()
        data.extend(data_json["results"])
    return data

def getreview(x):
    result = tmdb_movie.search(x)
    movie_id = result[0].id
    response = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}/reviews?api_key={tmdb.api_key}&language=en-US&page=1")
    data_json = response.json()
    return data_json

def getrating(title):
    movie_review = []
    data = getreview(title)
    for i in data['results']:
        pred = clt.predict(vectorizer.transform([i['content']]))
        movie_review.append({"review": i['content'], "rating": "Good" if pred[0] == 1 else "Bad"})
    return movie_review

def get_data2(x):
    result = tmdb_movie.search(x)
    movie_id = result[0].id
    trailer = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={tmdb.api_key}&language=en-US")
    response = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={tmdb.api_key}")
    return [response.json(), trailer.json()]

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/getname')
def getnames():
    return jsonify(df2["title_x"].tolist())

@app.route('/getmovie/<path:movie_name>')
def getmovie(movie_name):
    return jsonify(get_data2(movie_name))

@app.route('/getreview/<path:movie_name>')
def getreviews(movie_name):
    return jsonify(getrating(movie_name))

@app.route('/getdirector/<path:movie_name>')
def getdirectorname(movie_name):
    return jsonify(getdirector(movie_name))

@app.route('/getswipe')
def getswipe():
    return jsonify(get_swipe())

@app.route('/getnews')
def getnewsdata():
    return jsonify(get_news())

def get_recommendations(title, user_id):
    movies_data = pd.read_csv('Main_data.csv')
    ratings_data = pd.read_csv('movie_rating.csv')

    def content_based_recommendations(title, movies_data, top_n=6):
        movies_data['comb'] = movies_data['title_x'] + movies_data['genres']
        if title not in movies_data['title_x'].values:
            new_row = {'title_x': title, 'genres': ''}
            movies_data = pd.concat([movies_data, pd.DataFrame([new_row])], ignore_index=True)
        movies_data['comb'] = movies_data['comb'].fillna('')
        tfidf = TfidfVectorizer(stop_words='english')
        count_matrix = tfidf.fit_transform(movies_data['comb'])
        idx = movies_data[movies_data['title_x'] == title].index[0]
        cosine_sim = cosine_similarity(count_matrix, count_matrix)
        sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:top_n + 1]
        return movies_data['title_x'].iloc[[i[0] for i in sim_scores]]

    def collaborative_filtering_recommendations(user_id, top_n=10):
        merged_data = pd.merge(movies_data, ratings_data, left_on='id', right_on='movieId')
        user_item_matrix = pd.pivot_table(merged_data, values='rating', index='userId', columns='movieId', fill_value=0)
        if user_id not in user_item_matrix.index:
            print("User ID doesn't exist.")
            return None
        item_similarity = pairwise_distances(user_item_matrix.T, metric='cosine')
        min_rating, max_rating = user_item_matrix.min().min(), user_item_matrix.max().max()
        normalized = (user_item_matrix - min_rating) / (max_rating - min_rating)
        user_ratings = normalized.loc[user_id].values.reshape(1, -1)
        predicted = np.dot(user_ratings, item_similarity) / np.sum(item_similarity)
        predicted = predicted * (max_rating - min_rating) + min_rating
        indices = np.argsort(-predicted)[0][:top_n]
        return movies_data[movies_data['id'].isin(user_item_matrix.columns[indices])]['title_x']

    content_recs = content_based_recommendations(title, movies_data)
    collab_recs = collaborative_filtering_recommendations(user_id)
    if collab_recs is None:
        return content_recs
    return pd.concat([content_recs, collab_recs]).drop_duplicates()

@app.route('/send/<path:movie_name>/<string:userId>')
def get(movie_name, userId):
    print(f"Request received for movie: {movie_name}, userId: {userId}")
    val = get_recommendations(movie_name, userId)
    if val is None:
        return jsonify({"message": "movie or user not found in database"}), 404
    result = []
    for i in val:
        try:
            res = get_data2(i)
            result.append(res[0])
        except requests.ConnectionError:
            continue
    return jsonify(result)

@app.route('/rate/<movieId>/<rate>/<string:userId>')
def rate_movie(movieId, rate, userId):
    try:
        rate = float(rate)
    except ValueError:
        return jsonify({"error": "Invalid rating format"}), 400
    data = {'userId': userId, 'movieId': movieId, 'rating': rate}
    with open('movie_rating.csv', 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['userId', 'movieId', 'rating'])
        writer.writerow(data)
    return jsonify(data)

@app.route('/review/<movieId>/<path:review>/<string:userId>')
def review_movie(movieId, review, userId):
    data = {'userId': userId, 'movieId': movieId, 'review': review}
    with open('IMDB Dataset.csv', 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['userId', 'movieId', 'review'])
        writer.writerow(data)
    return jsonify(data)

@app.route('/store/<movieId>/<path:movie1>/<string:userId>')
def store_movie(movieId, movie1, userId):
    data = {'userId': userId, 'movieId': movieId, 'movie1': movie1}
    with open('Main_data.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data.values())
    return jsonify(data)

@app.route('/score/<path:title1>/<path:title2>/')
def findscore(title1, title2):
    title1, title2 = unquote(title1), unquote(title2)
    movies_data = pd.read_csv('Main_data.csv')

    # Create 'comb' if missing and fill NaN with empty string
    if 'comb' not in movies_data.columns:
        movies_data['comb'] = movies_data['title_x'] + movies_data['genres']
    movies_data['comb'] = movies_data['comb'].fillna('')

    # Dynamically fetch missing movies and update movies_data and CSV
    for title in [title1, title2]:
        if title not in movies_data['title_x'].values:
            try:
                result = tmdb_movie.search(title)
                if not result:
                    continue
                movie_id = result[0].id
                details = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={tmdb.api_key}").json()
                genre_names = ','.join([g['name'] for g in details.get("genres", [])])
                new_row = {
                    'title_x': title,
                    'genres': genre_names,
                    'comb': title + genre_names
                }

                # Append to CSV
                csv_file_path = 'Main_data.csv'
                file_exists = os.path.exists(csv_file_path)
                with open(csv_file_path, 'a', newline='', encoding='utf-8') as file:
                    writer = csv.DictWriter(file, fieldnames=['title_x', 'genres', 'comb'])
                    if not file_exists or os.stat(csv_file_path).st_size == 0:
                        writer.writeheader()
                    writer.writerow(new_row)

                # Append to DataFrame in memory
                movies_data = pd.concat([movies_data, pd.DataFrame([new_row])], ignore_index=True)
                movies_data['comb'] = movies_data['comb'].fillna('')

            except Exception as e:
                print(f"Error fetching or adding {title}: {e}")
                return jsonify({'error': f"Could not process movie: {title}"}), 404

    count_vec = CountVectorizer()
    count_matrix = count_vec.fit_transform(movies_data['comb'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    try:
        idx1 = movies_data[movies_data['title_x'] == title1].index[0]
        idx2 = movies_data[movies_data['title_x'] == title2].index[0]
    except IndexError:
        return jsonify({'error': 'One or both titles not found'}), 404

    sim = cosine_sim[idx1, idx2]

    tfidf = TfidfVectorizer(stop_words='english').fit(movies_data['comb'])
    features = tfidf.transform(movies_data['comb']).toarray()

    euc = euclidean_distances([features[idx1]], [features[idx2]])[0][0]
    man = manhattan_distances([features[idx1]], [features[idx2]])[0][0]

    return jsonify({
        'cosineSimilarity': round(sim, 4),
        'euclideanDistance': round(euc, 4),
        'manhattanDistance': round(man, 4)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5001)
