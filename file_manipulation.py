import pandas as pd
def read_tsv_input_to_df(filename):
    print("reading input file")
    return pd.read_csv(filename, delimiter='\t')

        

def output_trace(filename, data):
    print("outputing trace file")

def output_evaluation(filename, data):
    print("outputing evaluation file")