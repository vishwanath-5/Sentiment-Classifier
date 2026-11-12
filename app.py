import streamlit as st
import pickle 
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

def transfrom_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return ' '.join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Sentiment Classifier')

input_text = st.text_area('Enter the Text to Classify the Sentiment: ')

if st.button('Predict'):
    transformed_text = transfrom_text(input_text)
    vector_input = tfidf.transform([transformed_text])
    result = model.predict(vector_input)
    if result==0:
        st.text('Negative')
    elif result==2:
        st.text('Positive')
    elif result==1:
        st.text('Neutral')
    else:
        st.text('Neutral')