import pandas as pd
import math

FACTUAL_CLAIM_ID = 'q1_label'
FACTUAL_CLAIM_VALUE = "yes"
NOT_FACTUAL_CLAIM_VALUE = "no"
SMOOTHING_ADDITIVE = 0.01

class NaiveBayesClassifier:

    def __init__(self, dataset):
        self.total_size = len(dataset.index)
        self.vocabulary_size = len(dataset.columns)
        self.log_probability_factual = float()
        self.log_probability_not_factual = float()
        self.log_probablility_table_factual = None
        self.log_probability_table_not_factual = None
        self.dataset_contain_factual_claim = dataset[dataset[FACTUAL_CLAIM_ID] == FACTUAL_CLAIM_VALUE]
        self.dataset_contain_not_factual_claim = dataset[dataset[FACTUAL_CLAIM_ID] == NOT_FACTUAL_CLAIM_VALUE]

    def fit(self):
        self._set_pobability_of_factual_and_not()
        self._set_probablility_tables_for_factual_and_not()

    def _set_pobability_of_factual_and_not(self):
        self.log_probability_factual = math.log(len(
            self.dataset_contain_factual_claim.index)/self.total_size)
        self.log_probability_not_factual = math.log(len(
            self.dataset_contain_not_factual_claim.index)/self.total_size)

    def _set_probablility_tables_for_factual_and_not(self):
        self.log_probablility_table_factual = self._get_probablility_table(
            self.dataset_contain_factual_claim)
        self.log_probability_table_not_factual = self._get_probablility_table(
            self.dataset_contain_not_factual_claim)

    def _get_probablility_table(self, dataset):
        probability_table = dataset.drop(columns=[FACTUAL_CLAIM_ID]).sum(axis=0)
        total_number_of_words = probability_table.sum(axis=0) + (self.vocabulary_size * SMOOTHING_ADDITIVE)
        return probability_table.apply(lambda x: math.log((x + SMOOTHING_ADDITIVE) / total_number_of_words))

    def test(self, words):
        factual_probability = self._get_factual_claim_probability(words)
        not_factual_probability = self._get_not_factual_claim_probability(words)
        if factual_probability > not_factual_probability:
            return FACTUAL_CLAIM_VALUE, factual_probability
        return NOT_FACTUAL_CLAIM_VALUE, not_factual_probability

    def _get_factual_claim_probability(self, words):
        return self.log_probability_factual + self._get_attribute_probability(words, self.log_probablility_table_factual)

    def _get_not_factual_claim_probability(self, words):
        return self.log_probability_not_factual + self._get_attribute_probability(words, self.log_probability_table_not_factual)

    def _get_attribute_probability(self, words, probablility_table):
        probability = 0
        for word in words:
            probability = probability + probablility_table.loc[word]
        return probability
