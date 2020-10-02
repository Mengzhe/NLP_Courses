from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
import re
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
    # last 100 rows will be test
    Xtrain = X[:-100,]
    Ytrain = labels[:-100,]
    Xtest = X[-100:,]
    Ytest = labels[-100:,]


    model = MultinomialNB()
    model.fit(Xtrain, Ytrain)
    print("Classification rate for NB:", model.score(Xtest, Ytest))


    from sklearn.ensemble import AdaBoostClassifier

    model = AdaBoostClassifier()
    model.fit(Xtrain, Ytrain)
    print("Classification rate for AdaBoost:", model.score(Xtest, Ytest))



# vectorizer = CountVectorizer()
# vectorizer = TfidfVectorizer()

main(CountVectorizer())
print('')
main(TfidfVectorizer())