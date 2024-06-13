import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import string

import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import streamlit as st

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load your pre-trained model (example)
# Replace this with your actual model loading code
def load_model():
    model = LogisticRegression()  # Example model, replace with your model
    return model

# Text preprocessing function
def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def text_preprocessing(text):
    text_tokenize = word_tokenize(text)
    entity = text_tokenize[0]
    text_content = text_tokenize[1:]
    text_pos = pos_tag(text_content)
    remove_words = set(list(string.punctuation) + stopwords.words('english'))
    text_remove = [(word, pos) for (word, pos) in text_pos if word.lower() not in remove_words]
    word_lem = WordNetLemmatizer()
    text_lem = [(word_lem.lemmatize(word, pos=get_wordnet_pos(pos)) if get_wordnet_pos(pos) else word_lem.lemmatize(word), pos) for (word, pos) in text_remove]
    text_lem.append((entity,))
    return text_lem

# Load model and vectorizer
model = load_model()  # Replace with your model loading function

# Example training data
train_data = ["Positive Overwatch is a great game.", "Negative Overwatch is boring."]
train_labels = ["Positive", "Negative"]

# Combine text_preprocessing, CountVectorizer and TfidfTransformer into pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_preprocessing)),
    ('tfidf', TfidfTransformer()),
    ('classifier', model)
])

# Fit pipeline on the training data
pipeline.fit(train_data, train_labels)

# Streamlit App
st.title('Sentiment Analysis App')
st.write('Welcome to my sentiment analysis app!')

# Form to input text
st.subheader('Enter Text for Sentiment Analysis')
form = st.form(key='sentiment-form')
user_input = form.text_area('Enter your text')
submit_button = form.form_submit_button('Analyze Sentiment')

# Processing user input
if submit_button:
    if user_input:
        # Predict sentiment
        prediction = pipeline.predict([user_input])[0]
        
        # Display result
        st.subheader('Sentiment Prediction:')
        st.write(f'Text: {user_input}')
        st.write(f'Predicted Sentiment: {prediction}')
    else:
        st.warning('Please enter some text.')

# Footer or additional information
st.write('---')
st.write('Note: This is a simple example of sentiment analysis. Adjustments may be needed for your specific use case.')