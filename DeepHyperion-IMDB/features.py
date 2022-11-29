import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import re


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')

lemma = WordNetLemmatizer()
stop_words = stopwords.words('english')


file = open('opinion-lexicon-English/negative-words.txt', 'r')
neg_words = file.read().split()

file = open('opinion-lexicon-English/positive-words.txt', 'r')
pos_words = file.read().split()

def text_prep(x):
    corp = str(x).lower()
    corp = re.sub('[^a-zA-Z]+', ' ', corp).strip()
    tokens = word_tokenize(corp)
    words = [t for t in tokens if t not in stop_words]
    lemmatize = [lemma.lemmatize(w) for w in words]
    return lemmatize


def compute_sentiment(text):
    preprocess_text = text_prep(text)
    total_len = len(text)
    num_pos = len([i for i in preprocess_text if i in pos_words])
    num_neg = len([i for i in preprocess_text if i in neg_words])

    sentiment = round((num_pos - num_neg)/total_len , 2)
    # sentiment = round(num_pos/(num_neg+1),2)

    return sentiment

def count_pos(text):
    
    preprocess_text = text_prep(text)
    num_pos = len([i for i in preprocess_text if i in pos_words])
    return num_pos

def count_neg(text):
    preprocess_text = text_prep(text)
    num_neg = len([i for i in preprocess_text if i in neg_words])
    return num_neg

def count_pos_relative(text):
    preprocess_text = text_prep(text)

    
    num_pos = len([i for i in preprocess_text if i in pos_words])
    total = count_words(text)

    if total > 0:
        return int((num_pos/total)*100)
    else:
        return 0

def count_neg_relative(text):
    preprocess_text = text_prep(text)
    num_neg = len([i for i in preprocess_text if i in neg_words])
    total = count_words(text)
    if total > 0:
        return int((num_neg/total)*100)
    else:
        return 0



def compute_sentiment2(text):
    sent = SentimentIntensityAnalyzer()
    polarity = [round(sent.polarity_scores(i)['compound'], 2) for i in text]
    return polarity



def count_words(text):
    txt = text.split()
    count = 0
    for word in txt:
        if word not in [".", ",", ":", ";", "<", ">", "/", "!", "?"]:
            count += 1
    return count

    
def count_words_using_vec(text):
    from sklearn.feature_extraction.text import CountVectorizer
    # create the transform
    vectorizer = CountVectorizer()
    # tokenize and build vocab
    vectorizer.fit(text)
    # summarize
    print(vectorizer.vocabulary_)
    # encode document
    vector = vectorizer.transform(text)
    # summarize encoded vector
    print(vector.shape)
    print(type(vector))
    print(vector.toarray())
    return sum(vector)


def count_verbs(text):
    tokenized_text = word_tokenize(text)
    word_tags = nltk.pos_tag(tokenized_text)
    verbs = [wt[0] for wt in word_tags if wt[1] in ['VB', 'VBD', 'VBN', 'VBP', 'VBZ', 'VBG']]
    return len(verbs)


def count_adjs(text):
    tokenized_text = word_tokenize(text)
    word_tags = nltk.pos_tag(tokenized_text)
    adjs_advs = [wt[0] for wt in word_tags if wt[1] in ['JJ', 'JJR', 'JJS']]
    return len(adjs_advs)



