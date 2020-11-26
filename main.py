from file_manipulation import read_tsv_input_to_df
import string
import pandas as pd
import re

from NaiveBaysClass.NaiveBayesClassifier import NaiveBayesClassifier

_vocabulary = {}
_filtered_vocabulary = {}
_text_vectors=[]
_filtered_text_vectors=[]
_smoothing =0.01
#Log 10 
def build_vocabulary(training_file_name):
    global _vocabulary
    global _filtered_vocabulary
    global _text_vectors
    global _filtered_text_vectors
    # nlp = spacy.load('en_core_web_sm')
    training_set = read_tsv_input_to_df(training_file_name)
    training_set['text'] = training_set['text'].str.lower()

    training_set['text'].apply(count_words,vocab=_vocabulary)
    training_set['text'].apply(count_words,vocab=_filtered_vocabulary)
    _filtered_vocabulary = {key:val for key,val in _filtered_vocabulary.items() if val != 1}

    # print(_filtered_vocabulary.keys())
    # df = pd.DataFrame.from_dict(_vocabulary)
    # print(df)
    print(len(_filtered_vocabulary))
    #BOW reguler
    training_set['text'].apply(build_BOW,vocab=list(_vocabulary.keys()),master_vect=_text_vectors)
    df_general = pd.DataFrame(data=_text_vectors)
    df_general.columns=list(_vocabulary.keys())

    df_general['q1_label'] = training_set['q1_label']
    model = NaiveBayesClassifier(df_general)
    model.fit()
    validated_array = ["self", "quarantine", "#coronavirus", ]
    print(model.test(validated_array))
    # print(df_general)


    # BOW filtered
    training_set['text'].apply(build_BOW,vocab=list(_filtered_vocabulary.keys()),master_vect=_filtered_text_vectors)
    df_filtered = pd.DataFrame(data=_filtered_text_vectors)
    df_filtered.columns=list(_filtered_vocabulary.keys())

    # build_BOW(training_set['text'][0],vocab=list(_vocabulary.keys()),master_vect=_text_vectors)
    # print(_text_vectors)
    # print(_vocabulary)
    # print(_filtered_vocabulary)

    # print(training_set['text'][0])
    # training_set.apply(str.lower(), columns=['text'])




def build_BOW(text,vocab,master_vect):
    list_words = re.split(r'\s+',text)
    text_vect = [0]*len(vocab)
    for word in vocab:
        if word in list_words:
            text_vect[vocab.index(word)] += list_words.count(word)
    master_vect.append(text_vect)



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