from flask import Flask, render_template, request, jsonify
import pickle
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from neattext.functions import remove_punctuations, remove_special_characters, remove_stopwords, remove_numbers, remove_shortwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
# Chargement du modèle OneVsRest SVM
with open('models/SVMClassifier.pickle', 'rb') as f:
    model = pickle.load(f)

# Chargement du TF-IDF Vectorizer
with open('models/TFIDFVectorizer.pickle', 'rb') as f:
    vectorizer = pickle.load(f)

tag_names = ['.NET',
 '.NET-CORE',
 'AMAZON-WEB-SERVICES',
 'ANDROID',
 'ANDROID-STUDIO',
 'ANGULAR',
 'APACHE-SPARK',
 'ARRAYS',
 'ASP.NET-CORE',
 'AZURE',
 'C',
 'C#',
 'C++',
 'CSS',
 'DART',
 'DATAFRAME',
 'DEEP-LEARNING',
 'DJANGO',
 'DOCKER',
 'EXPRESS',
 'FIREBASE',
 'FLUTTER',
 'GIT',
 'HTML',
 'IOS',
 'JAVA',
 'JAVASCRIPT',
 'JQUERY',
 'JSON',
 'KERAS',
 'KOTLIN',
 'KUBERNETES',
 'LARAVEL',
 'LINUX',
 'MACHINE-LEARNING',
 'MACOS',
 'NEXT.JS',
 'NODE.JS',
 'NPM',
 'NUMPY',
 'PANDAS',
 'PHP',
 'POSTGRESQL',
 'PYTHON',
 'PYTHON-3.X',
 'R',
 'REACT-NATIVE',
 'REACTJS',
 'SPRING',
 'SPRING-BOOT',
 'SQL',
 'SWIFT',
 'SWIFTUI',
 'TENSORFLOW',
 'TYPESCRIPT',
 'VISUAL-STUDIO-CODE',
 'VUE.JS',
 'WEBPACK',
 'WINDOWS',
 'XCODE']

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
                               output = get_tag_from_list(predictions[0], tag_names),
                               user_title = question_title,
                               user_body = question_body)
  

def lemmatization(list_of_words):
    """
    Transform words into lemmas
    
    Args:
        list_of_words(list): List of words
        
    Returns:
        lemmatized(list): List of lemmatized words
    """
    
    lemmatizer = WordNetLemmatizer()
    lemmatized = []
    
    for word in list_of_words:
        lemmatized.append(lemmatizer.lemmatize(word.lower()))
        
    return lemmatized

def preprocess_text(text):
    # Lowercase the text
    cleaned_text = text.lower()
    # Remove punctuations
    cleaned_text = remove_punctuations(cleaned_text)
    # Remove special characters
    cleaned_text = remove_special_characters(cleaned_text)
    # Remove stopwords
    cleaned_text = remove_stopwords(cleaned_text)
    # Remove numbers
    cleaned_text = remove_numbers(cleaned_text)
    # Remove short words
    cleaned_text = remove_shortwords(cleaned_text)
    # Tokenization
    tokenized_text = word_tokenize(cleaned_text)
    # Lemmatization
    lemmatized_text = lemmatization(tokenized_text)
    # Convert tokenized lemmatized text back to string
    preprocessed_text = " ".join(lemmatized_text)
    return preprocessed_text

def get_tag_from_list(tags_list, tag_names):
    if len(tags_list) != len(tag_names):
        return "Error: Invalid list length!"
    
    selected_tags = []
    for i in range(len(tags_list)):
        if tags_list[i] == 1:
            selected_tags.append(tag_names[i])
    
    if len(selected_tags) == 0:
        return "No tags predicted."
    
    return ', '.join(selected_tags)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
