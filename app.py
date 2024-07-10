import streamlit as st
import pickle
import requests
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies=pd.read_csv('updated_data.csv')
cv=CountVectorizer(max_features=5000,stop_words='english',lowercase=True)
vectors=cv.fit_transform(movies['tags']).toarray()

similarity=cosine_similarity(vectors)

def fetch_movie_poster(movie_id):
    api_key = 'df83397c2ada702406a527ccd574ef80'
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?language=en-US&api_key={api_key}"
    response = requests.get(url)
    data=response.json()
    # st.text(url)
    return "https://image.tmdb.org/t/p/w500/"+data['poster_path']


def recommend(movie):
    movie_index=movies[movies['title']==movie].index[0]
    distances=similarity[movie_index]
    top_5_movies_tuple=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    recommended_movies=[]
    recommended_movies_poster=[]
    for i in top_5_movies_tuple:
        movie_id=movies.iloc[i[0]].movie_id  #440
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_movies_poster.append(fetch_movie_poster(movie_id))
    return recommended_movies,recommended_movies_poster

# similarity=pickle.load(open('similarities.pkl','rb'))
movies=pickle.load(open('movies.pkl','rb'))
movies_list=movies['title'].values

st.title('Movie Recommendation System')

selected_movie_name = st.selectbox(
    "Enter the movies that you want to watch",
    movies_list)

if st.button("Recommend"):
    top_5_movies_recommended,top_5_movies_recommended_posters=recommend(selected_movie_name)
    col1, col2, col3, col4, col5 = st.columns(5)

# Column 1
    with col1:
        st.text(top_5_movies_recommended[0])
        st.image(top_5_movies_recommended_posters[0])

# Column 2
    with col2:
        st.text(top_5_movies_recommended[1])
        st.image(top_5_movies_recommended_posters[1])

# Column 3
    with col3:
        st.text(top_5_movies_recommended[2])
        st.image(top_5_movies_recommended_posters[2])

# Column 4
    with col4:
        st.text(top_5_movies_recommended[3])
        st.image(top_5_movies_recommended_posters[3])

# Column 5
    with col5:
        st.text(top_5_movies_recommended[4])
        st.image(top_5_movies_recommended_posters[4])

      
      