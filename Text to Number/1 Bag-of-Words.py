# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 20:39:40 2023

@author: Sibo Ding
"""

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

def bow(filename, df_stopwords):
    '''
    Bag-of-words
    Input a ".txt" file (filename without ".txt")
    Output a dataframe with the number of occurrences of each word
    '''
    
    # Read ".txt" file
    f = open(filename + '.txt', 'r')
    content = f.read()
    f.close()

    # Change to lower case; Tokenization
    # https://towardsdatascience.com/5-simple-ways-to-tokenize-text-in-python-92c6804edfc4
    df_token = pd.DataFrame(word_tokenize(content.lower()), columns=['word'])

    # Only retain words start with letter using regular expression
    # This is robust after checking several financial statements
    df_pure_words = pd.DataFrame()
    df_pure_words['word'] = df_token['word'].str.extract('(^[a-z].*)').dropna()

    # Remove stop words (filter words that are not in stop words)
    df_no_stop = \
        df_pure_words[~df_pure_words['word'].isin(df_stopwords['word'])]

    # Stemming and lemmatization
    # https://www.datacamp.com/tutorial/stemming-lemmatization-python
    df_stem = pd.DataFrame(
        [PorterStemmer().stem(i) for i in df_no_stop['word']],
        columns=['word'])
    
    df_lemma = pd.DataFrame(
        [WordNetLemmatizer().lemmatize(i) for i in df_stem['word']],
        columns=['word'])
    
    # Bag-of-words: Count the number of occurrences of each word
    df_lemma[filename] = 1
    df_bow = df_lemma.groupby('word').count().reset_index()
    return df_bow


# Stop words
# https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
df_stopwords = pd.DataFrame(stopwords.words('english'), columns=['word'])

# Combine vectors of each observation into a whole matrix
df_bow_all = pd.DataFrame(columns=['word'])

for year in range(1994, 2002+1):
    for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
        filename = str(year) + quarter
        df_bow = bow(filename, df_stopwords)
        df_bow_all = pd.merge(df_bow_all, df_bow, how='outer', on='word')

# Transpose and set each word as column names
df_transpose = df_bow_all.transpose()
df_transpose.columns = df_transpose.iloc[0]
df_transpose = df_transpose.drop(df_transpose.index[0]).reset_index(names='quarter_statement')
# Fill NaN by 0
df_fill_na = df_transpose.fillna(0)

# Export ".csv" file
df_fill_na.to_csv('bag_of_words.csv', index=False)
