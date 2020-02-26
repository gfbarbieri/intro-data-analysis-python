### Term Frequency Index Document Frequency and ngram based string matching
### see comments in analysis()

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from scipy import sparse
import csv
import os

# n-grams
def ngrams(string, n=2):
    string = (re.sub(r"[^\sA-Za-z0-9]", "", string)).upper() 
    # remove all special characters
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

# distance measure between name A and name B; simply A dot B in matrix
def cosine_measure(A, B, ntop=1, lower_bound=0):
    A = A.tocsr()
    B = B.tocsr()
    temp = (A * B).toarray()
    temp2 = []
    # convert sparse matrix to regular-format matrix
    for row in temp:
        row = row.tolist()
        max_row = row.index(max(row))
        temp_row = [0] * len(row)
        temp_row[max_row] = max(row)
        temp2 += [temp_row]

    return sparse.csr_matrix(temp2)

def get_matches_df(sparse_matrix, A, B, top=100):
    non_zeros = sparse_matrix.nonzero()

    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]

    if top:
        nr_matches = top
    else:
        nr_matches = sparsecols.size

    left_side = np.empty([nr_matches], dtype=object)
    right_side = np.empty([nr_matches], dtype=object)
    similairity = np.zeros(nr_matches)

    for index in range(0, nr_matches):
        left_side[index] = A[sparserows[index]]
        right_side[index] = B[sparsecols[index]]
        similairity[index] = sparse_matrix.data[index]

    return pd.DataFrame({'raw_data_df_raw': left_side,
                         'matched_data_df_clean': right_side,
                         'cosine_measure': similairity})

def analysis():

    # ### your list that needs to match one in df_clean
    # df_raw = ["gogle","microsoft bing","amazn","facebuk","fcbook","gogle INC",
    #           "bing INC","ms bing","amazn INC","faebokk","fcbook INC.",
    #           "alphabet google","the facebook", "the facebok INC"]

    # ### replace df_clean by your cleaned list
    # df_clean = ["google", "bing", "amazon", "facebook"]

    ## see an example below. I use Alrinton's tpops and LDB data. 
    ## input files: tpops.csv and ldb.csv
    ## output file: ouotput.csv

    path = os.getcwd()
    df = pd.read_csv(os.path.join(path, 'tpops.csv'))
    df_raw = list(set(df.OLTNAME.values)) # get elements in OLTNAME column of tpop.csv as df_raw


    df2 = pd.read_csv(os.path.join(path, 'ldb.csv'))
    df_clean_o1 = df2.LEGAL_NAME.values
    df_clean_o2 = df2.TRADE_NAME.values
    df_clean = df_clean_o1 + df_clean_o2 # use both LEGAL_NAME and TRADE_NAME in LDB table for a matching process
    #print(df_clean)


    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams) ## use ngram=2. it looks to me that n=2 works best for name matching process.
    tf_idf_matrix_clean = vectorizer.fit_transform(df_clean)
    tf_idf_matrix_raw = vectorizer.transform(df_raw)
    matches = cosine_measure(tf_idf_matrix_raw, tf_idf_matrix_clean.transpose(), 1, 0)
    matches_df = get_matches_df(matches, df_raw, df_clean, top=0)

    # ### if you want to show the result on screen. commment out this.
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(matches_df.sort_values(by=['cosine_measure'], ascending=False))

    ### save the result as 'output.csv'
    outpuffilename = 'output.csv'
    matches_df.sort_values(by=['cosine_measure'], ascending=False).to_csv(outpuffilename, index= False )

analysis()