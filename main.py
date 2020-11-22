from file_manipulation import read_tsv_input_to_df
import string

import re

_vocabulary = {}
_filtered_vocabulary = {}
_smoothing =0.01
#Log 10 
def build_vocabulary(training_file_name):
    global _vocabulary
    global _filtered_vocabulary
    # nlp = spacy.load('en_core_web_sm')
    training_set = read_tsv_input_to_df(training_file_name)
    training_set['text'] = training_set['text'].str.lower()
    training_set['text'].apply(count_words,vocab=_vocabulary)
    training_set['text'].apply(count_words,vocab=_filtered_vocabulary)
    _filtered_vocabulary = {key:val for key,val in _filtered_vocabulary.items() if val != 1}

    print(_vocabulary)
    print(_filtered_vocabulary)

    # print(training_set['text'])
    # training_set.apply(str.lower(), columns=['text'])

def count_words(text, vocab):
    # regex_punc = map(str,string.punctuation)
    list_words = re.split(r'\s+',text)
    for word in list_words:
        if word in vocab:
            vocab[word] +=1
        else:
            vocab[word] = 1

if __name__ == "__main__":
    build_vocabulary('covid_training.tsv')