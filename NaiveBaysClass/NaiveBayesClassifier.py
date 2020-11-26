import pandas as pd
import math

FACTUAL_CLAIM_ID = 'q1_label'
FACTUAL_CLAIM_VALUE = "yes"
NOT_FACTUAL_CLAIM_VALUE = "no"
SMOOTHING_ADDITIVE = 0.01

class NaiveBayesClassifier:

    def __init__(self, dataframe, vocabulary_size):
        self.dataset = dataframe
        self.vocabulary_size = vocabulary_size
        self.log_prob_factual_claim = float()
        self.log_prob_not_factual_claim = float()
        self.log_probablility_table_factual_claim = 0
        self.log_probability_table_not_factual_claim = 0
        self.dataset_contain_factual_claim = self.dataset[self.dataset[FACTUAL_CLAIM_ID] == FACTUAL_CLAIM_VALUE]
        self.dataset_not_contain_factual_claim = self.dataset[self.dataset[FACTUAL_CLAIM_ID] == NOT_FACTUAL_CLAIM_VALUE]

    def fit(self):
        self._set_pobability_of_contain_or_not_factual_claim()
        self._set_probablility_tables_for_factual_and_not()

    def _set_pobability_of_contain_or_not_factual_claim(self):
        total_size = len(self.dataset.index)
        self.log_prob_factual_claim = math.log(len(
            self.dataset_contain_factual_claim.index)/total_size)
        self.log_prob_not_factual_claim = math.log(len(
            self.dataset_not_contain_factual_claim.index)/total_size)

    def _set_probablility_tables_for_factual_and_not(self):
        self.log_probablility_table_factual_claim = self._get_probablility_table(
            self.dataset_contain_factual_claim)
        self.log_probability_table_not_factual_claim = self._get_probablility_table(
            self.dataset_not_contain_factual_claim)

    def _get_probablility_table(self, dataset):
        probability_table = dataset.drop(
            columns=[FACTUAL_CLAIM_ID]).sum(axis=0)
        # NOT SURE IF WE MULTIPLY THE VOCAB SIZE BY THE SMOOTHING AND THEN ADD IT OR JUST ADD THE VOCAB SIZE
        total_number_of_words = probability_table.sum(axis=0) + (self.vocabulary_size * SMOOTHING_ADDITIVE)
        return probability_table.apply(lambda x: math.log((x + SMOOTHING_ADDITIVE) / total_number_of_words))

    def test(self, words):
        if self._get_factual_claim_probability(words) >= self._get_not_factual_claim_probability(words):
            return FACTUAL_CLAIM_VALUE
        return NOT_FACTUAL_CLAIM_VALUE

    def _get_factual_claim_probability(self, words):
        return self.log_prob_factual_claim + self._get_attribute_probability(words, self.log_probablility_table_factual_claim)

    def _get_not_factual_claim_probability(self, words):
        return self.log_prob_not_factual_claim + self._get_attribute_probability(words, self.log_probability_table_not_factual_claim)

    def _get_attribute_probability(self, words, probablility_table):
        probability = 0
        for word in words:
            probability = probability + probablility_table.loc[word]
        return probability