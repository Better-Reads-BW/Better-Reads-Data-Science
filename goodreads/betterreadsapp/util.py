from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
import pandas as pd
import numpy as np
import string
import re

tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()
tfidf_vectorizer = TfidfVectorizer()
stemmer = PorterStemmer()
stopwords = '\nltk_data\corpora\stopwords\english'
# stopwords = '\nltk_data\corpora\wordnet\english'

# df_original = pd.read_csv('book_data.csv')
df = pd.read_csv('preprocessed_df.csv')
documents = np.array(df['book_desc'])
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

def preprocessor(text):
    new_text = re.sub('<.*?>', '', text)   # remove HTML tags
    new_text = re.sub("[!@#$+%*:()'-]",'',new_text) # remove punc.
    new_text = re.sub(r'\d+','',new_text)# remove numbers
    new_text = new_text.lower() # lower case, .upper() for upper
    tokenized_text = tokenizer.tokenize(new_text)
    words = [w for w in tokenized_text if w not in stopwords]
    lem_text = [lemmatizer.lemmatize(i) for i in words]
    stem_text = " ".join([stemmer.stem(i) for i in lem_text])
    return stem_text

def predictions(text):
    documents = [preprocessor(text)]
    tfidf_matrix_new = tfidf_vectorizer.transform(documents)
    array = cosine_similarity(tfidf_matrix_new, tfidf_matrix)[0]
    recommender = df.copy()
    recommender['cs'] = array
    recommender.sort_values(by=['cs'], ascending=False)
    return recommender.nlargest(5, 'cs')