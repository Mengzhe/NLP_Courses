# https://deeplearningcourses.com/c/data-science-natural-language-processing-in-python
# https://www.udemy.com/data-science-natural-language-processing-in-python

# Author: http://lazyprogrammer.me

# Get the data from here:
# https://lazyprogrammer.me/course_files/moby_dick.txt

### encode a message

# this is a random excerpt from Project Gutenberg's
# The Adventures of Sherlock Holmes, by Arthur Conan Doyle
# https://www.gutenberg.org/ebooks/1661

# original_message = '''I then lounged down the street and found,
# as I expected, that there was a mews in a lane which runs down
# by one wall of the garden. I lent the ostlers a hand in rubbing
# down their horses, and received in exchange twopence, a glass of
# half-and-half, two fills of shag tobacco, and as much information
# as I could desire about Miss Adler, to say nothing of half a dozen
# other people in the neighbourhood in whom I was not in the least
# interested, but whose biographies I was compelled to listen to.
# '''


original_message = "I like cat"

import re
from collections import Counter
def preprocessing(message):
    ## remove all non-alphanumeric characters
    message = re.sub(r'\W+', ' ', message)
    message = message.split(' ')
    return message

def build_bigram_prob(message):
    bigram_prob = Counter()
    for word in message:
        n = len(word)
        for i in range(n-1):
            bigram_prob[word[i:i+2]] += 1
    return bigram_prob

def build_unigram_prob(message):
    unigram_prob = Counter()
    for word in message:
        n = len(word)
        for i in range(n):
            unigram_prob[word[i]] += 1
    return unigram_prob

def bulid_initial_char_prob(message):
    initial_char_prob = Counter()
    for word in message:
        initial_char_prob[word[0]] += 1
    return initial_char_prob

## language model
message = preprocessing(original_message)
# print(message)
bigram_prob = build_bigram_prob(message)
unigram_prob = build_unigram_prob(message)
initial_char_prob = bulid_initial_char_prob(message)
# print(bigram_prob)
# print(unigram_prob)
# print(initial_char_prob)


