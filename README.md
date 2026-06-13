# Movie-Recommender-system
🎥 A Machine Learning -based Movie Recommender System that suggests personalised movie using Content-Based Filtering.
# 🎬 Movie Recommender System

A personalized Movie Recommender System built with Python that suggests movies based on user preferences. 

## 📌 Features
* **Search Functionality:** Type a movie name and get instant recommendations.
* **Poster Fetching:** Automatically fetches real-time movie posters using the TMDB API.

## 🛠️ Tools & Technologies Used
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn (Cosine Similarity)

## 🧠 How It Works
This system primarily uses **Content-Based Filtering**. It analyzes movie metadata (genres, keywords, cast, crew, and overview) to create tags. Using **Cosine Similarity**, the algorithm calculates the distance between movie vectors to recommend the top 5 closest matches to the user's selection.

