from file_manipulation import read_tsv_input_to_df
import pandas as pd
import re
import glob
import os
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


def build_vocabularies():
    global _vocabulary
    global _filtered_vocabulary
    global _text_vectors
    global _filtered_text_vectors

    training_set['text'].apply(count_words, vocab=_vocabulary)
    training_set['text'].apply(count_words, vocab=_filtered_vocabulary)
    _filtered_vocabulary = {key: val for key, val in _filtered_vocabulary.items() if val != 1}



def build_BOW(text, vocab, master_vect):
    list_words = re.split(r'\s+', text)
    text_vect = [0]*len(vocab)
    for word in vocab:
        if word in list_words:
            text_vect[vocab.index(word)] += list_words.count(word)
    master_vect.append(text_vect)


def count_words(text, vocab):
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


def run_naive_bay(dataframe, test_file_name, vocab, type):

    print("Building model...")
    model = NaiveBayesClassifier(dataframe)
    model.fit()
    print("Done building...")
    print("Running test on " + type)
    test_set = read_tsv_input_to_df(test_file_name)
    test_set['text'] = test_set['text'].str.lower()

    test_tweets ={}
    test_set.apply(sanitize_tweet, vocabulary=vocab,test_tweets=test_tweets, axis=1)

    number_of_correct = 0
    number_of_wrong = 0
    tp_yes = 0
    tp_no = 0
    fp_yes = 0
    fp_no = 0
    fn_yes = 0
    fn_no = 0

    for id, tweet in test_tweets.items():
        actual_value = test_set.loc[test_set['tweet_id'] == id]['q1_label'].values[0]
        test_value, probability = model.test(tweet)
        correctness = 'wrong'
        if actual_value == test_value:
            if actual_value == 'yes':
                tp_yes+=1
            elif actual_value == 'no':
                tp_no+=1
            correctness = 'correct'
            number_of_correct += 1
        else:
            if actual_value == 'yes':
                if test_value == 'no':
                    fn_yes+=1
                    fp_no+=1
            elif actual_value == 'no':
                if test_value == 'yes':
                    fp_yes+=1
                    fn_no+=1
            number_of_wrong += 1
        trace_output(id,test_value,probability,actual_value,correctness,type)

    eval_output(tp_yes,fp_yes,fn_yes,tp_no,fp_no,fn_no,number_of_correct,number_of_wrong,type)
    print("Done evaluation and trace")

def calc_precision(tp, fp):
    return tp/(tp+fp)

def calc_recall(tp,fn):
    return tp/(tp+fn)

def calc_f1(p,r):
    return (2*p*r)/(p+r)

def trace_output(id, likely_label, score, actual, quality,type):
    with open('trace_NB-BOW-'+type+'.txt','a+') as f:
        f.write(str(id) +'  ' +str(likely_label)+'  ' + str("{:.2e}".format(score))+'  ' + str(actual) +'  '+ str(quality)+'\n')


def eval_output(tp_yes,fp_yes,fn_yes,tp_no,fp_no,fn_no,correct,wrong,type):
    with open('eval_NB-BOW-'+type+'.txt','w') as f:
        f.write(
            str(round(correct/(wrong+correct),4)) + '\n' +
            str(round(calc_precision(tp_yes,fp_yes),4)) + '  ' + str(round(calc_precision(tp_no,fp_no),4)) + '\n' +
            str(round(calc_recall(tp_yes,fn_yes),4)) + '  ' + str(round(calc_recall(tp_no,fn_no),4)) + '\n' +
            str(round(calc_f1(calc_precision(tp_yes,fp_yes),calc_recall(tp_yes,fn_yes)),4)) + '  ' + str(round(calc_f1(calc_precision(tp_no,fp_no),calc_recall(tp_no,fn_no)),4)) + '\n'
        )


def clean_dir():
    files = glob.glob('trace_NB-BOW-*.txt')
    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print("Couldn't remove file "+ f)

if __name__ == "__main__":
    clean_dir()
    set_up('covid_training.tsv')
    build_vocabularies()
    run_naive_bay(get_BOW_regular(), 'covid_test_public.tsv',list(_vocabulary.keys()),'OV')
    print('running filtered')
    run_naive_bay(get_BOW_filtered(), 'covid_test_public.tsv', list(_filtered_vocabulary.keys()),'FV')
