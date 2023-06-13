# =[Modules dan Packages]========================

from flask import Flask,render_template,request,jsonify
import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
nltk.download("stopwords")
nltk.download('wordnet')
import pickle
import os
from tensorflow.keras.models import load_model

# =[Variabel Global]=============================

app   = Flask(__name__, static_url_path='/static')

# Load the model
loaded_model = load_model('Bidirectional LSTM English EMO.h5')


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
        
stop_words=set(stopwords.words('english'))
def preprocess(text):
    # Normalization
    sentence = re.sub('@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+', ' ', text)
    sentence = sentence.lower()
    sentence = sentence.split()
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

	# Run Flask di localhost 
	#app.run(host="localhost", port=5000, debug=True)
	app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 8080)))

