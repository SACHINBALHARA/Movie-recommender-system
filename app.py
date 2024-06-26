import os
import streamlit as st
import streamlit_extras
import pandas as pd
import numpy as np
import sklearn
import pickle
import gzip
import shutil
st.title("Welcome to the Movie Recommender System")


# Decompress the similarity file
with gzip.open('similarity.pkl.gz', 'rb') as f_in:
    with open('similarity.pkl', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

# Load the similarity matrix and the dataset
with open('similarity.pkl', 'rb') as file:
    similarity_matrix = pickle.load(file)


# load the pretrained model
with open("preprocessed_dataset.pkl",'rb') as file:
    model=pickle.load(file)


# Function to recommend movies
def recommend(movie):
    try:
        index = model[model['title'] == movie].index[0]
        distances = sorted(list(enumerate(similarity_matrix[index])), reverse=True, key=lambda x: x[1])
        recommended_movies = [model.iloc[i[0]].title for i in distances[1:6]]
        return recommended_movies
    except IndexError:
        return ["Movie not found in the dataset."]



# user input
movie_name=st.text_input("enter movie name you like:")

if st.button('Recommend'):
    if movie_name:
        recommendations = recommend(movie_name)
        st.write("Recommended Movies:")
        for movie in recommendations:
            st.write(movie)
    else:
        st.write("Please enter a movie name.")
    




