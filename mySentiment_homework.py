"""
My own solution to the sentiment problem. 10/12/2020
"""

import nltk
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier

from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup

## https://www.geeksforgeeks.org/python-lemmatization-with-nltk/
wordnet_lemmatizer = WordNetLemmatizer()

# from http://www.lextek.com/manuals/onix/stopwords1.html
# stopwords = set(w.rstrip() for w in open('stopwords.txt'))

# note: an alternative source of stopwords
from nltk.corpus import stopwords

# nltk.download('stopwords')
stopwords = stopwords.words('english')
# print(stopwords)

# nltk.download('punkt') ## for word_tokenize
# nltk.download('wordnet') ## for wordnet_lemmatizer.lemmatize

import tensorflow as tf

# load the reviews
# data courtesy of http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html
positive_reviews = BeautifulSoup(open('electronics/positive.review').read(), features="html5lib")
positive_reviews = positive_reviews.findAll('review_text')

negative_reviews = BeautifulSoup(open('electronics/negative.review').read(), features="html5lib")
negative_reviews = negative_reviews.findAll('review_text')

ls_positive_reviews = [''.join(positive_reviews[i].contents) for i in range(len(positive_reviews))]
ls_negative_reviews = [''.join(negative_reviews[i].contents) for i in range(len(negative_reviews))]


## customized tokenizer
def tokenize(reviews):
    reviews = [nltk.tokenize.word_tokenize(line) for line in reviews]
    res = []
    for line in reviews:
        new_line = []
        for i in range(len(line)):
            if len(line[i]) <= 2: continue
            new_line.append(wordnet_lemmatizer.lemmatize(line[i].lower()))

        res.append(new_line.copy())

    return res

tokenized_pos_reviews = tokenize(ls_positive_reviews)
tokenized_neg_reviews = tokenize(ls_negative_reviews)
# print(tokenized_pos_reviews[0])
# print(tokenized_neg_reviews[0])

word_index = {}
cur_index = 0
def getWordIndex(review, start_idx):
    cur_index = start_idx
    for line in review:
        for word in line:
            if word not in word_index:
                word_index[word] = cur_index
                cur_index += 1


## generate word index map from positive and negative reviews
getWordIndex(tokenized_pos_reviews, 0)
# print(len(word_index))
getWordIndex(tokenized_neg_reviews, len(word_index))
# print(len(word_index))
# print(word_index)
vocab_size = len(word_index)

def vectorize(tokens, label):
    x = np.zeros(vocab_size + 1)  ## the last is the label
    for token in tokens:
        idx = word_index[token]
        x[idx] += 1
    x = x / x.sum()  # normalize it before setting label
    x[-1] = label
    return x

N = len(tokenized_pos_reviews) + len(tokenized_neg_reviews)
# (N x D+1 matrix - keeping them together for now so we can shuffle more easily later
data = np.zeros((N, vocab_size+1))
# print(data.shape)
i = 0
for tokenized_review in tokenized_pos_reviews:
    x = vectorize(tokenized_review, 1)
    data[i, :] = x
    i += 1
for tokenized_review in tokenized_neg_reviews:
    x = vectorize(tokenized_review, 0)
    data[i, :] = x
    i += 1

X = data[:, :-1]
Y = data[:, -1]
# print(data.shape, X.shape, Y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

model = LogisticRegression()
model.fit(X_train, y_train)
print("Result using Logistic Regression")
print("Train accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))

model = AdaBoostClassifier()
model.fit(X_train, y_train)
print("Result using AdaBoost Classifier")
print("Train accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))