import requests

def test_recommendations(movie_name, user_id):
    url = f"http://localhost:5001/send/{movie_name}/{user_id}"
    print(f"Requesting recommendations for movie: '{movie_name}', user: '{user_id}'\n")
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if not data:
            print("No recommendations found.")
        else:
            print(f"Received {len(data)} recommendations:")
            for i, movie in enumerate(data, 1):
                title = movie.get("title", "Unknown Title")
                rating = movie.get("vote_average", "N/A")
                print(f"{i}. {title} (Rating: {rating})")
    except Exception as e:
        print(f"Error during API request: {e}")

if __name__ == "__main__":
    # Replace these with valid movie name and user ID from your dataset
    movie_name = "Avatar"  # example movie title
    user_id = "H8MuqLVgoAWvi7d2bnCtQVkgSNz1"  # example user ID

    # URL encode movie name if it contains spaces or special characters
    from urllib.parse import quote
    movie_name_encoded = quote(movie_name)

    test_recommendations(movie_name_encoded, user_id)
