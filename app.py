# base library
import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask
import pickle
from keras.models import load_model
from flask import Flask, request, jsonify
import json
import os


# Preprocessing
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =[Variabel Global]=============================

app   = Flask(__name__, static_url_path='/static')

# Load the model
loaded_model = load_model('Bidirectional LSTM English EMO.h5')
# try to load stop words
#nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


# # [Routing untuk Halaman Utama atau Home]	
# @app.route("/")
# def beranda():
#     return render_template('index.html')

# [Routing untuk API]		
@app.route("/api/deteksi",methods=['POST'])
def apiDeteksi():
	# Nilai default untuk string input s
	text_input = ""
	
	if request.method=='POST':
		# Set nilai string input dari pengguna
		print(text_input)
		text_input = request.form['data']
		
		# Text Pre-Processing
		text_input = preprocess(text_input)
                
		# Prediksi hasil
		# loading
		with open('label_encoder.pkl', 'rb') as handle:
			le = pickle.load(handle)
		result = le.inverse_transform(np.argmax(loaded_model.predict(text_input), axis=-1))[0]
		proba = np.max(loaded_model.predict(text_input))
		print(f"{result} : {proba}\n\n")

		# Return hasil prediksi dengan format JSON
		return jsonify({
			"data": result,
		})
        
# stop_words=set(stopwords.words('english'))
# def lemmatization(text):
#     doc = nlp(text)
#     return " ".join([token.lemma_ for token in doc])
# stop_words=set(stopwords.words('english'))
with open('stop_words.pkl', 'rb') as handle:
     stop_words = pickle.load(handle)
# def lemmatization(text):
# 	doc = nlp(text)
# 	return " ".join([token.lemma_ for token in doc])

def preprocess(text):
    # Normalization
    sentence = re.sub('@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+', ' ', text)
    sentence = sentence.lower()
    sentence = sentence.split()
    # sentence = lemmatization(sentence)
    sentence = [word for word in sentence if not word in stop_words]
    # Loading Tokenizer
    with open('tokenizer.pkl', 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)
    # Tokenization
    sentence = loaded_tokenizer.texts_to_sequences([sentence])
    # Padding
    sentence = pad_sequences(sentence, maxlen=229, truncating='pre')
    return sentence

# =[Main]========================================

if __name__ == '__main__':
	app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 8080)))

