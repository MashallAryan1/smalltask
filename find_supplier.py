#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import ast
import sklearn
from sklearn.neighbors import NearestNeighbors    
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
import argparse

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 



def read_invoice(file_name):
    """
    Reads a file of invoice content created by OCR and converts it to a pandas dataframe
    
    :param file_name: Name of the file containing invoice content
    :return: A dataframe of the invoice words 
    
    """
    with open(file_name,'r') as f:
        words = "[{}]".format(f.read().strip())
    words = words.replace("\n", ",")
    words = ast.literal_eval(words)
    inv_df = pd.DataFrame(words)
    # sort the words according to their order in the original doc
    inv_df = inv_df.set_index(['page_id','line_id','pos_id'])
    inv_df.sort_index(inplace=True)
    
    return ' '.join(inv_df['word'])



    
lemat = WordNetLemmatizer()
def clean(item):
    """
    preprocessing the input string to remove the unwanted characters and substrings     
    :param item: input string
    :return: a string in which unwanted characters and substrings are removed 
    
    """
    res = item.encode("ascii", errors="ignore").decode()
    res = re.sub('['+string.punctuation+']+','',res).strip()
    res = word_tokenize(res.lower())
    res = [lemat.lemmatize(item) for item in res if item not in stopwords.words('english')]    
    return   " ".join(res)

# def ngram(doc, n=(3,4)):
#
#     seq = doc.split()
#     res = []
#     for num_gram in range(n[0],n[1]+1):
#         for i in range(len(seq)-num_gram):
#             res+=[' '.join(seq[i:i+num_gram])]
#     return res
    
    
def main(args):
    inv_file = args['invoice']
    sup_file = args['suppliers']
    
    # read suppliers names
    sup_df = pd.read_csv(sup_file)
    sup_df['SupplierName'] = sup_df['SupplierName'].apply(clean)
    
    
    # produce tfidf features
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(ngram_range=(1, 4), analyzer='word')
    doc_term_matrix = vectorizer.fit_transform(sup_df['SupplierName'])
    
    nearestnbr = NearestNeighbors(n_neighbors=1).fit(doc_term_matrix ) 
    
    query = read_invoice(inv_file)
    
    query = clean(query)

    query_tfidf = vectorizer.transform(query.split())
    
    distances, indices = nearestnbr.kneighbors(query_tfidf)
    suppliers_index = indices[distances.argmin()]
    supplier = sup_df.iloc[suppliers_index]
    return supplier['Id'].values[0], supplier['SupplierName'].values[0]



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--invoice',
                        type=str, required=True,
                        help="Path to the file of invoice words.")
    parser.add_argument('-s', '--suppliers',
                        type=str, default='suppliernames.txt',
                        help="Path to the file of supplier names.")

    args = parser.parse_args()
    inv_file = args.invoice
    sup_file = args.suppliers
    print( main({'invoice':inv_file,'suppliers':sup_file}) )
