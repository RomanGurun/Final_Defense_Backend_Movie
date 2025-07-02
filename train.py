# train.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import pickle

# ✅ More realistic movie review data (60 samples total: 30 pos, 30 neg)
positive_reviews = [
    "Absolutely loved this movie. Brilliant acting and script.",
    "An inspiring story told in a beautiful way.",
    "The movie had stunning visuals and a powerful message.",
    "Great performances. I was emotionally moved.",
    "This film is a masterpiece. Every scene is memorable.",
    "A perfect mix of drama and thrill. Highly recommend.",
    "The soundtrack and acting were both top-notch.",
    "A heartwarming story. Brought tears to my eyes.",
    "Cinematography was amazing. Beautifully shot.",
    "A must-watch for anyone who appreciates great filmmaking.",
    "Incredible story, brilliantly executed.",
    "One of the best movies I’ve seen this year.",
    "This film exceeded all expectations.",
    "Truly a remarkable piece of cinema.",
    "Emotional and gripping. A true work of art.",
    "Outstanding performance by the lead actor.",
    "It was a visual and emotional treat.",
    "Great direction and tight screenplay.",
    "Loved the narrative style and flow.",
    "The plot was original and engaging.",
    "Powerful dialogues and impactful scenes.",
    "Each character was beautifully portrayed.",
    "Wonderful pacing and storytelling.",
    "Masterful work from the director.",
    "Great chemistry between the leads.",
    "Smart writing and clever plot twists.",
    "Amazing acting and a fantastic ending.",
    "A rollercoaster of emotions and suspense.",
    "Beautifully crafted and deeply emotional.",
    "A fresh perspective and strong acting."
]

negative_reviews = [
    "Terrible movie. I walked out halfway.",
    "Worst movie I’ve ever seen. Total waste of time.",
    "Plot made no sense and acting was bad.",
    "This film was a disaster in every aspect.",
    "Painfully slow and boring. Do not recommend.",
    "The script was weak and full of clichés.",
    "Disappointing. I expected much more.",
    "A dull and forgettable experience.",
    "Very poor character development.",
    "The plot was predictable and lame.",
    "Absolutely no depth or originality.",
    "Cinematography was the only good thing.",
    "Dialogue felt forced and unrealistic.",
    "Storyline was messy and confusing.",
    "Completely failed to hold my interest.",
    "This film lacked soul and creativity.",
    "Horrible direction and editing.",
    "Acting felt flat and uninspired.",
    "Overhyped and underwhelming.",
    "Soundtrack didn't fit the scenes.",
    "Unbelievably bad performances.",
    "I regret watching this movie.",
    "Couldn't wait for it to end.",
    "A snooze-fest from beginning to end.",
    "Weak plot, poor execution.",
    "Pointless film with zero engagement.",
    "Tries too hard but fails badly.",
    "Confusing and poorly written.",
    "Just plain bad. Nothing good to say.",
    "Unintentionally hilarious for the wrong reasons."
]

# Combine and label
reviews = positive_reviews + negative_reviews
labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)

# Create DataFrame
df = pd.DataFrame({'review': reviews, 'sentiment': labels})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print("📊 Accuracy:", accuracy)
print("📄 Classification Report:\n", classification_report(y_test, y_pred))

# Save model and vectorizer
with open("vectorizerer.pkl", "wb") as f_vec:
    pickle.dump(vectorizer, f_vec)

with open("nlp_model.pkl", "wb") as f_model:
    pickle.dump(model, f_model)

print("✅ Saved: 'vectorizerer.pkl' and 'nlp_model.pkl'")
