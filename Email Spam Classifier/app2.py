import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Ensure necessary NLTK data files are downloaded
nltk.download('stopwords')

# Initialize the Porter Stemmer
ps = PorterStemmer()

# Load the vectorizer and Naive Bayes model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Function for text preprocessing
def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize
    text = [word for word in text if word.isalnum()]  # Remove non-alphanumeric characters
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]  # Remove stopwords and punctuation
    text = [ps.stem(word) for word in text]  # Apply stemming
    return " ".join(text)

# Streamlit code
st.title("Email Spam Classifier")

# Input field for SMS message
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # Preprocess the input message
    transformed_sms = transform_text(input_sms)
    
    # Vectorize the preprocessed message
    vector_input = tfidf.transform([transformed_sms])
    
    # Predict
    result = model.predict(vector_input)[0]
    
    # Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
