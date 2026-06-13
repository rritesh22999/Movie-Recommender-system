import numpy as np
import pandas as pd
# 1. Force Pandas to show EVERY column without collapsing them
pd.set_option('display.max_columns', None)

# 2. Expand the display width so it doesn't wrap awkwardly 
pd.set_option('display.width', 1000)

movies = pd.read_csv(r'C:\Users\rrite\OneDrive\Desktop\movie recommended system\tmdb_5000_movies.csv')
credits = pd.read_csv(r'C:\Users\rrite\OneDrive\Desktop\movie recommended system\tmdb_5000_credits.csv')

# 1. See the first 5 rows of the movies dataset
print("--- MOVIES HEAD ---")
print(movies.head())

# 2. See all the column names available in movies
print("\n--- MOVIES COLUMNS ---")
print(movies.columns.tolist())

# 3. See the first 5 rows of the credits dataset to check it too
print("\n--- CREDITS HEAD ---")
print(credits.head())
print(credits.shape)
print(movies.shape)
movies=movies.merge(credits,on='title')
print(movies.head(1))
print(movies.info())
#columns to keep
# genres,id,keywords,title(original title we won't keep as movies title may be in native lang.),overview,cast,crew
movies=movies[['movie_id','title','genres','keywords','overview','cast','crew']]
print(movies.head())
movies.dropna(inplace=True)
print(movies.isnull().sum())#OVERVIEW ME 3 MOVIES AIYSE H, JINKA OVERVIEW HUME NHI PTA SO DROP THEM
print(movies.duplicated().sum())
print(movies.iloc[0].genres)
#{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}

import ast
def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name']) 
    return L

#ast.literal_eval({"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"})
###### ast.literal_eval use?
#convert({"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}), data stotred in form of dictonary
movies['genres']=movies['genres'].apply(convert)
print(movies.head())
# now the genres column contains the info of (action,adventure,fantasy..)

movies['keywords']=movies['keywords'].apply(convert)
#print(movies.head())

def convert3(obj):
    L=[]
    count=0
    for i in ast.literal_eval(obj):
        if count!= 3:
            L.append(i['name']) 
            count+=1
        else:
            break    
    return L

movies['cast']=movies['cast'].apply(convert3)
print(movies['cast'])
#print(movies.head())

def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']== 'Director':  # here in director use uppercase otherwise its creates problem
            L.append(i['name']) 
            break
    return L

movies['crew']=movies['crew'].apply(fetch_director)
print(movies['crew'])
#print(movies.head())

# overview section is a string, make it in a list to concatinate properly
movies['overview']=movies['overview'].apply(lambda x:x.split())
print(movies['overview'][0]) # overview ka 0th row print kar ke dekha kaisa output aa rha h
print(movies.head())
movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
print(movies['genres'])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])
print(movies.head())  ##########  finally now our data is ready to worked, now continate

movies['tags']= movies['overview']+ movies['genres']+ movies['keywords']+ movies['cast']+ movies['crew']

new_df=movies[['movie_id','title','tags']]
print(new_df)
new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))
print(new_df.head())

new_df['tags']=new_df['tags'].apply(lambda x:x.lower())
print(new_df['tags'][0])


##############   count vectoriser  
# from sklearn.feature_extraction.text import CountVectorizer
# cv = CountVectorizer(max_features=5000,stop_words='english')
# vectors = cv.fit_transform(new_df['tags']).toarray()
# print(vectors[0])
# print(cv.get_feature_names_out())

###########  stemming   

import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)    
print(stem ('in the 22nd century, a paraplegic marine is dispatched to the moon pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. action adventure fantasy sciencefiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d samworthington zoesaldana sigourneyweaver jamescameron'))

new_df['tags']=new_df['tags'].apply(stem)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
print(vectors[0])
print(cv.get_feature_names_out()[0:100])

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)
print(similarity[0])  # 1st movie ki 4806 movies se similarity score dega ye
print(similarity[1])

############  shorting karenge(reverse order m) and usme se top 5 movies extract kr lenge but problem ye aayegi ki index value like- koi bhi movie phle aa sakti h, khoo jayega to usko hume counter karna padega  #########  enumerate function ka use karenge  
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity [movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)

        
# 1. Ask the user for a movie and save their answer in a variable
user_movie = input("Enter a movie name: ")

# 2. Pass that variable into your function!
recommend(user_movie)