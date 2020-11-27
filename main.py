from os import curdir
from file_manipulation import read_tsv_input_to_df
import string
import pandas as pd
import re

from NaiveBaysClass.NaiveBayesClassifier import NaiveBayesClassifier

_vocabulary = {}
_filtered_vocabulary = {}
_text_vectors = []
_filtered_text_vectors = []

training_set = None


def set_up(training_file_name):
    global training_set
    training_set = read_tsv_input_to_df(training_file_name)
    training_set['text'] = training_set['text'].str.lower()

# Log 10


def build_vocabulary():
    global _vocabulary
    global _filtered_vocabulary
    global _text_vectors
    global _filtered_text_vectors
    # nlp = spacy.load('en_core_web_sm')

    training_set['text'].apply(count_words, vocab=_vocabulary)
    training_set['text'].apply(count_words, vocab=_filtered_vocabulary)
    _filtered_vocabulary = {key: val for key, val in _filtered_vocabulary.items() if val != 1}

    # print(_filtered_vocabulary.keys())
    # df = pd.DataFrame.from_dict(_vocabulary)
    # print(df)
    # print(len(_filtered_vocabulary))
    # BOW reguler
    # training_set['text'].apply(build_BOW,vocab=list(_vocabulary.keys()),master_vect=_text_vectors)
    # df_general = pd.DataFrame(data=_text_vectors)
    # df_general.columns=list(_vocabulary.keys())

    # df_general['q1_label'] = training_set['q1_label']
    # model = NaiveBayesClassifier(df_general)
    # model.fit()
    # validated_array = ["self", "quarantine", "#coronavirus", ]
    # print(model.test(validated_array))
    # print(df_general)

    # BOW filtered
    # training_set['text'].apply(build_BOW,vocab=list(_filtered_vocabulary.keys()),master_vect=_filtered_text_vectors)
    # df_filtered = pd.DataFrame(data=_filtered_text_vectors)
    # df_filtered.columns=list(_filtered_vocabulary.keys())

    # build_BOW(training_set['text'][0],vocab=list(_vocabulary.keys()),master_vect=_text_vectors)
    # print(_text_vectors)
    # print(_vocabulary)
    # print(_filtered_vocabulary)

    # print(training_set['text'][0])
    # training_set.apply(str.lower(), columns=['text'])


def build_BOW(text, vocab, master_vect):
    list_words = re.split(r'\s+', text)
    text_vect = [0]*len(vocab)
    for word in vocab:
        if word in list_words:
            text_vect[vocab.index(word)] += list_words.count(word)
    master_vect.append(text_vect)


def count_words(text, vocab):
    # regex_punc = map(str,string.punctuation)
    list_words = re.split(r'\s+', text)
    for word in list_words:
        if word in vocab:
            vocab[word] += 1
        else:
            vocab[word] = 1


def get_BOW_regular():
    # BOW reguler
    training_set['text'].apply(build_BOW, vocab=list(
        _vocabulary.keys()), master_vect=_text_vectors)
    df_general = pd.DataFrame(data=_text_vectors)
    df_general.columns = list(_vocabulary.keys())
    df_general['q1_label'] = training_set['q1_label']
    return df_general


def get_BOW_filtered():
    # BOW filtered
    training_set['text'].apply(build_BOW, vocab=list(
        _filtered_vocabulary.keys()), master_vect=_filtered_text_vectors)
    df_filtered = pd.DataFrame(data=_filtered_text_vectors)
    df_filtered.columns = list(_filtered_vocabulary.keys())
    df_filtered['q1_label'] = training_set['q1_label']
    return df_filtered


def sanitize_tweet(dataframe, vocabulary, test_tweets):
    sanitised_tweet = list()
    list_words = re.split(r'\s+', dataframe['text'])
    for word in list_words:
        if word in vocabulary:
            sanitised_tweet.append(word)
    test_tweets[dataframe['tweet_id']] = sanitised_tweet


def run_naive_bay(dataframe, test_file_name, vocab):

    print("Building model...")
    model = NaiveBayesClassifier(dataframe)
    model.fit()
    print("Done building...")

    test_set = read_tsv_input_to_df(test_file_name)
    test_set['text'] = test_set['text'].str.lower()

    test_tweets ={}
    test_set.apply(sanitize_tweet, vocabulary=vocab,test_tweets=test_tweets, axis=1)

    number_of_correct = 0
    number_of_wrong = 0 
    for id, tweet in test_tweets.items():
        actual_value = test_set.loc[test_set['tweet_id'] == id]['q1_label'].values[0]
        test_value, probability = model.test(tweet)
        correctness = 'wrong'
        if actual_value == test_value:
            correctness = 'correct'
            number_of_correct += 1
        else:
            number_of_wrong += 1

            
        print(str(id) +'  ' +str(test_value)+'  ' + str(probability)+'  ' + str(actual_value) +'  '+ str(correctness))
        # trace_output(id,model.test(tweet),"1e6",test_set.loc[test_set['tweet_id']==id])
    print("number of correct: "+str(number_of_correct))
    print("number of wrong: "+str(number_of_wrong))

def trace_output(id, likely_label, score, actual, quality):
    print(id)


if __name__ == "__main__":
    set_up('covid_training.tsv')
    build_vocabulary()
    run_naive_bay(get_BOW_regular(), 'covid_test_public.tsv',list(_vocabulary.keys()))
    print('running filtered')
    run_naive_bay(get_BOW_filtered(), 'covid_test_public.tsv', list(_filtered_vocabulary.keys()))
