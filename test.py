
import csv
import itertools
import operator
import numpy as np
import nltk
import sys
import os
import time
from datetime import datetime
#from utils import *
#from rnn_theano import RNNTheano
#from VanillaRNN import *
import string
import pickle


vocabulary_size = 2000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

LOAD_DATASET = False

def getData(file):
    # Read the data and append SENTENCE_START and SENTENCE_END tokens
    print "Reading CSV file..."
    sentences = []
    with open(file, 'rb') as f:
        reader = csv.reader(f, skipinitialspace=True)
        reader.next()
        # Split full comments into sentences
        for x in reader:
            if len(x) > 0:
                c = x[0].decode('utf-8').lower()
                sentences.append(c)

        # Append  SENTENCE_END

        sentences = ["%s %s" % (x, sentence_end_token) for x in sentences]

    print "Parsed %d sentences." % (len(sentences))

    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print "Found %d unique words tokens." % len(word_freq.items())

    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(vocabulary_size - 1)
    index2word = [x[0] for x in vocab]
    index2word.append(unknown_token)
    word2index = dict([(w, i) for i, w in enumerate(index2word)])

    num_sentences_with_each_word = num_sentences_with_unique_words(vocab, sentences)

    print "Using vocabulary size %d." % vocabulary_size
    print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])

    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word2index else unknown_token for w in sent]

    print "\nExample sentence: '%s'" % sentences[0]
    print "\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0]

    # Create the training data
    training_data = np.asarray([[word2index[w] for w in sent] for sent in tokenized_sentences])

    return [training_data, word2index, index2word, vocab, num_sentences_with_each_word]


def load_dataset(train_file_en, train_file_de):
    training_data = None
    word2index = None
    index2word = None
    vocab = None
    num_sentences_with_each_word = None


    if LOAD_DATASET:
        pass
        dataset_file = 'translate_dataset.p'

        with open(dataset_file, 'rb') as f:
            dataset = pickle.load(f)

            train_data_X = dataset['en']
            train_data_Y = dataset['de']

    else:

        dataset = dict()

        train_data_X = getData(train_file_en)
        train_data_Y = getData(train_file_de)

        dataset['en'] = train_data_X
        dataset['de'] = train_data_Y

        save_dataset(dataset)

    return train_data_X, train_data_Y


def shrink_file(no_of_lines=5000, input=None, output=None):
    i = 0
    with open(input) as f:
        with open(output, "w") as f1:
            for line in f:
                if i < no_of_lines:
                    f1.write(line.replace(","," "))
                    i += 1
                else:
                    return


def get_training_set():

    """

    train_X = load_dataset(train_file_en)


    train_Y = getData(train_file_de)

    """
    train_file_en = 'europarl-en-small'
    train_file_de = 'europarl-de-small'


    train_X, train_Y = load_dataset(train_file_en, train_file_de)

    print ('Done tokenization!')
    return [train_X, train_Y]

def num_sentences_with_unique_words(vocab, sentences):
    num_sentences = []

    for i in range(len(vocab)):
        word = vocab[i][0]
        sent_count = 0
        for sentence in sentences:
            if find_substring(word, sentence):
                sent_count += 1

        num_sentences.append((word,sent_count))

    return num_sentences


def num_sentences_with_unique_words2(vocab, sentences, word2index, index2word, training_X):
    num_sentences = []

    for i in range(len(vocab)):
        word = vocab[i][0]
        sent_count = 0
        for sentence in training_X:
            for idx in sentence:
                if idx == word2index[word]:
                    sent_count += 1




def find_substring(word, sentence):

    index = sentence.find(word)
    if index != -1 and (ord(word[0]) == 46 or ord(word[0]) == 59):
        return True

    if index == -1:
        return False
    if index != 0 and sentence[index-1] not in string.whitespace:
        return False
    L = index + len(word)
    if L < len(sentence) and sentence[L] not in string.whitespace:
        return False
    return True


def save_dataset(data):
    pickle.dump(data, open("translate_dataset.p", "wb"))

def get_frquencies(y, vocab, num_sent_with_word):

    pass

def get_frquencies(file):
    import re
    frequency = {}


    document_text = open(file, 'r')
    text_string = document_text.read().lower()
    match_pattern = re.findall(r'\b[a-z]{1,15}\b | \b,\b | \b\.\b | \b\;\b | \b\"\b | '
                               r' \b\"\b | \b\?\b | \b\!\b', text_string)

    for word in match_pattern:
        count = frequency.get(word, 0)
        frequency[word] = count + 1

    frequency_list = frequency.keys()

    for words in frequency_list:
        print words, frequency[words]


get_training_set()



