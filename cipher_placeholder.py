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

import math
import random
import re
from collections import Counter
from string import ascii_lowercase

original_message = "I like cat"

def preprocessing(message):
    ## remove all non-alphanumeric characters
    message = re.sub(r'\W+', ' ', message)
    message = message.lower()
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

def comp_log_prob(word):
    ## with add_one smoothing
    c0 = word[0]
    init_prob = (initial_char_prob[c0] + 1) / (total_init + V_init)
    log_prob = math.log(init_prob)
    for i in range(1, len(word)):
        bigram = word[i-1:i+1]
        log_prob += math.log((bigram_prob[bigram]+1)/(unigram_prob[word[i]]+V_bigram))
    return log_prob

def genMapFromList(ls):
    m = {}
    for idx, c in enumerate(ascii_lowercase):
        m[c] = chr(ord('a') + ls[idx])
    return m

def get_random_maps(n):
    res = []
    for _ in range(n):
        ls = list(range(26))
        random.shuffle(ls)
        # m = genMapFromList(ls)
        # res.append(m)
        res.append(ls)
    return res

def decode(text, m):
    res = []
    for c in text:
        res.append(m[c])
    return ''.join(res)

def create_offsprings(ls, n):
    res = []
    for _ in range(n):
        temp = ls.copy()
        idx_0, idx_1 = random.choices(list(range(26)), k=2)
        temp[idx_0], temp[idx_1] = temp[idx_1], temp[idx_0]
        res.append(temp)
    return res




## language model
message = preprocessing(original_message)
print(message)
bigram_prob = build_bigram_prob(message)
unigram_prob = build_unigram_prob(message)
initial_char_prob = bulid_initial_char_prob(message)

V_bigram = 26*26
V_unigram = 26
V_init = 26
## denominator for computing x(0)
total_init = 0
for _, v in initial_char_prob.items():
    total_init += v


# print("bigram_prob", bigram_prob)
# print("unigram_prob", unigram_prob)
# print("initial_char_prob", initial_char_prob)
# tests_strings = ['la', 'li', 'il', 'ia', 'ik', 'i', 'a', 'xyz', 'abc', 'cat', 'lik', 'like', 'catt', 'njhg']
# for s in tests_strings:
#     print(s, comp_log_prob(s))
#
# for m in get_random_maps(5):
#     print(m)

ans_word = 'cat'
# ans_map = genMapFromList(list(range(26)))
# ans_ls = list(range(26))
# random.shuffle(ans_ls)
ans_ls = [16, 13, 5, 9, 22, 3, 11, 20, 8, 2, 18, 7, 4, 24, 1, 0, 15, 6, 25, 14, 19, 10, 21, 12, 23, 17]
# print(ans_ls)
ans_map = genMapFromList(ans_ls)
ans_word_decoded = decode(ans_word, ans_map)
print("ans_word", ans_word, "ans_word_decoded", ans_word_decoded)

# for ls in create_offsprings(list(range(26)), 3):
#     print(ls)






