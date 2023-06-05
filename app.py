import numpy as np
from flask import Flask, render_template, request, jsonify
import pickle
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from neattext.functions import remove_punctuations, remove_special_characters, remove_stopwords, remove_numbers, remove_shortwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Chargement du modèle OneVsRest SVM
with open('models/SVMClassifier.pickle', 'rb') as f:
    model = pickle.load(f)

# Chargement du TF-IDF Vectorizer
with open('models/TFIDFVectorizer.pickle', 'rb') as f:
    vectorizer = pickle.load(f)

app = Flask(__name__)

@app.route('/', methods = ["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template('index.html')
    if request.method == "POST":
        question_title = request.form.get("question-title")
        question_body = request.form.get("question-body")
        corpus = question_body + question_title

        preprocessed_corpus = preprocess_text(corpus)

        # Vectorisation des données
        vectorized_data = vectorizer.transform([preprocessed_corpus])

        # Prédiction avec le modèle
        predictions = model.predict(vectorized_data)

        # Renvoyer les prédictions sous forme de réponse JSON
        return render_template("index.html",
                               output = str(predictions),
                               user_title = question_title,
                               user_body = question_body)
  

if __name__ == '__main__':
    app.run()
