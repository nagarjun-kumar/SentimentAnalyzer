import nltk
import re
import string
from nltk import FreqDist
from nltk import classify
from nltk import NaiveBayesClassifier
import json
nltk.download('twitter_samples')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

from nltk.corpus import twitter_samples


text = twitter_samples.strings('tweets.20150430-223406.json')
pos_tokenized_tweets = twitter_samples.tokenized('positive_tweets.json')
neg_tokenized_tweets = twitter_samples.tokenized('negative_tweets.json')
sarcastic_tweets = 'Sarcasm_Headlines_Dataset_v2.json'

from nltk.tokenize import word_tokenize
import json
import re

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[09]|[-$_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][-a-z'\_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
    ]

tokens_re = re.compile(r"("+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)


def tokenize(s):
    return tokens_re.findall(s)


def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

def final_token():
    sarcastic_list = []
    with open('final.json') as f:
        data = json.load(f)
        for i in data['tweets']:
            tokens = preprocess(i['tweet'])
            for val in tokens:
                sarcastic_list.append(val)
    f.close()
    return sarcastic_list

from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer

def lemmatize(token):
    lemmatizer = WordNetLemmatizer()
    lemmatized_val = []

    for word, tag in pos_tag(token):
        
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_val.append(lemmatizer.lemmatize(word,pos))
    return lemmatized_val

sarc_list=final_token()

def remove_noise(tokenized_tweet, stop_words = ()):
    new_token = []
    for token, tag in pos_tag(tokenized_tweet):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)
        
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token,pos)
        
        if len(token)>0 and token not in string.punctuation and token.lower not in stop_words:
            new_token.append(token.lower())
    return new_token

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

#print(remove_noise(tokenized_tweet[0], stop_words))
pos_list = []
neg_list = []
#sarc_list = []
final_sarc_list = []
for val in pos_tokenized_tweets:
    pos_list.append(remove_noise(val, stop_words))
for val in neg_tokenized_tweets:
    neg_list.append(remove_noise(val, stop_words))
for val in sarc_list:
    final_sarc_list.append(remove_noise(val, stop_words))
    
def get_word_density(list_token):
    for values in list_token:
        for value in values:
            yield value
all_pos_words = get_word_density(pos_list)

freq_distrib_pos = FreqDist(all_pos_words)
#print(freq_distrib_pos.most_common(10))
    

def finalized_tweets(list_token):
    for values in list_token:
        yield dict([value, True] for value in values)

pos_model = finalized_tweets(pos_list)
neg_model = finalized_tweets(neg_list)
sarc_model = finalized_tweets(final_sarc_list)

import random

pos_dataset = [(tweet_dict, "Positive") for tweet_dict in pos_model]
neg_dataset =[(tweet_dict, 'Negative') for tweet_dict in neg_model]
sarc_dataset = [(tweet_dict, "Sarcastic") for tweet_dict in sarc_model]
finaldataset = pos_dataset+neg_dataset+sarc_dataset
random.shuffle(finaldataset)
train_data = finaldataset[:7000]
test_data = finaldataset[7000:]

our_classifier = NaiveBayesClassifier.train(train_data)
print("Accuracy is",  classify.accuracy(our_classifier,test_data))

from nltk.tokenize import word_tokenize
our_tweet =  'Today is A Beautiful Day'
our_token = remove_noise(word_tokenize(our_tweet))
print(our_classifier.classify(dict([value,True] for value in our_token)))

