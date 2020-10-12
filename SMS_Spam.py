from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
import re
from sklearn.utils import shuffle
from sklearn.ensemble import AdaBoostClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# corpus = [
#     'This is the first document.',
#     'This document is the second document.',
#     'And this is the third one.',
#     'Is this the first document?',
# ]

data = pd.read_csv('spam.csv', encoding='ISO-8859-1') # use pandas for convenience
# print(data[:3])
# print(type(data))

labels = data['v1']
raw_text = data['v2']

text = []
for line in raw_text:
    line = re.sub(r'[^A-Za-z0-9 ]+', '', line)
    text.append(line)
    # print(line)


def main(vectorizer):
    print("feature extractor", vectorizer.__class__.__name__)
    X = vectorizer.fit_transform(text)
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    Y = le.transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    model = MultinomialNB()
    model.fit(X_train, y_train)
    print("Classification rate for NB:", model.score(X_test, y_test))



    model = AdaBoostClassifier()
    model.fit(X_train, y_train)
    print("Classification rate for AdaBoost:", model.score(X_test, y_test))



# vectorizer = CountVectorizer()
# vectorizer = TfidfVectorizer()

main(CountVectorizer(decode_error='ignore'))
print('')
main(TfidfVectorizer(decode_error='ignore'))
