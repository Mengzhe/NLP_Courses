from collections import defaultdict
import random
import nltk
import re
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

# load the reviews
# data courtesy of http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html
positive_reviews = BeautifulSoup(open('electronics/positive.review').read(), features="html5lib")
positive_reviews = positive_reviews.findAll('review_text')

# negative_reviews = BeautifulSoup(open('electronics/negative.review').read(), features="html5lib")
# negative_reviews = negative_reviews.findAll('review_text')

ls_positive_reviews = [''.join(positive_reviews[i].text.lower()) for i in range(len(positive_reviews))]
# ls_negative_reviews = [''.join(negative_reviews[i].text.lower()) for i in range(len(negative_reviews))]
# print(ls_positive_reviews[0])

## tokenize
tokenized_pos_reviews = []
for review in ls_positive_reviews:
    review = re.sub(r'[^A-Za-z0-9 ]+', '', review)
    tokens = nltk.tokenize.word_tokenize(review)
    tokenized_pos_reviews.append(tokens)
# print(tokenized_pos_reviews[0])

trigram_counts = defaultdict(lambda: defaultdict(int))
total_counts = defaultdict(int)
for tokens in tokenized_pos_reviews:
    for i in range(1, len(tokens) - 1):
        trigram_counts[(tokens[i - 1], tokens[i + 1])][tokens[i]] += 1
        total_counts[(tokens[i - 1], tokens[i + 1])] += 1

trigram_model = defaultdict(lambda: defaultdict(float))
for context in trigram_counts:
    for word in trigram_counts[context]:
        trigram_model[context][word] = trigram_counts[context][word] / total_counts[context]

print(trigram_model)


## given a context (word_i-1, word i+1), sample a word from trigram_model
## based on the probability (weight) of each word
def random_sample_from_trigram_model(trigram_model, context):
    samples = list(trigram_model[context].items())
    rand_ = random.random()
    # print(samples)
    # print(rand_)
    cum_weight = 0  ## cumulative weight
    for word, weight in samples:
        cum_weight += weight
        if rand_ < cum_weight:
            return word

# rand_word = random_sample_from_trigram_model(trigram_model, ('this', 'stand'))
# print("rand_word", rand_word)

replace_prob = 0.2 ## probability to replace a word
review = random.choice(positive_reviews)
review = review.text.lower()
org_tokens = nltk.tokenize.word_tokenize(review)
res = [org_tokens[0]] ## res: thw tokens with replacement
for i in range(1, len(org_tokens)-1):
    ## make a replacement
    if random.random() < replace_prob:
        context = (org_tokens[i-1], org_tokens[i+1])
        if context in trigram_model:
           rep_word = random_sample_from_trigram_model(trigram_model, context)
           res.append(rep_word)
           print("context", context, "original", org_tokens[i], "replacement", rep_word)
    else:
        res.append(org_tokens[i])

res.append(org_tokens[-1])

print(org_tokens)
print(res)
