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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import streamlit as st

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Data Loading
kolom_name = ['tweet id', 'entity', 'sentiment', 'tweet content']
df = pd.read_csv("twitter_validation.csv", names=kolom_name, header=None)

# Fungsi pemrosesan teks
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

# Load models
def load_models():
    models = [
        ("Logistic Regression", LogisticRegression()),
        ("KNeighbors Classifier", KNeighborsClassifier()),
        ("Decision Tree Classifier", DecisionTreeClassifier()),
        ("Random Forest Classifier", RandomForestClassifier())
    ]
    return models

models = load_models()

# Data latih contoh
train_data = df['tweet content'].tolist()
train_labels = df['sentiment']

# Membuat pipeline untuk setiap model
pipelines = [(name, Pipeline([
    ('bow', CountVectorizer(analyzer=text_preprocessing)),
    ('tfidf', TfidfTransformer()),
    ('classifier', model)
])) for name, model in models]

# Melatih pipeline pada data latih
for name, pipeline in pipelines:
    pipeline.fit(train_data, train_labels)

# Aplikasi Streamlit
st.title('Aplikasi Analisis Sentimen')
st.write('Selamat datang di aplikasi analisis sentimen saya!')

# Form untuk memasukkan teks
st.subheader('Masukkan Teks untuk Analisis Sentimen')
form = st.form(key='sentiment-form')
user_input = form.text_area('Masukkan teks Anda')
model_choice = form.selectbox('Pilih Model', [name for name, model in models])
submit_button = form.form_submit_button('Analisis Sentimen')

# Memproses input pengguna
if submit_button:
    if user_input:
        # Memilih pipeline berdasarkan pilihan model
        selected_pipeline = dict(pipelines)[model_choice]

        # Memprediksi sentimen
        prediction = selected_pipeline.predict([user_input])[0]
        
        # Menampilkan hasil
        st.subheader('Prediksi Sentimen:')
        st.write(f'Teks: {user_input}')
        st.write(f'Sentimen yang Diprediksi: {prediction}')
    else:
        st.warning('Silakan masukkan teks.')

# Footer atau informasi tambahan
st.write('---')
st.write('Catatan: Ini adalah contoh sederhana analisis sentimen. Penyesuaian mungkin diperlukan untuk kasus penggunaan spesifik Anda.')
