#coding:utf-8
import nltk,json
from nltk import corpus
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt
plt.style.use('ggplot')
import pandas as pd
import os
import urllib
import numpy as np
import re
from unidecode import unidecode
#import xlrt,xlrd

df = pd.read_csv(os.getcwd()+'/df2.csv')
def getcontect(imdbid):
    try:
        url = "https://api.themoviedb.org/3/find/tt{:}?api_key=bb3beb7ec7af6d1c0c23ca7381b62a89&external_source=imdb_id".format(imdbid)
        response = requests.get(url)
        path = json.loads(response.text)
        context_text = path['movie_results'][0]['overview']    
    except: 
        context_text = ""
    return context_text

def preprocess(str_in):
    str_in = str_in.lower()
    return re.sub('[!"\'(),./?:\-]', ' ', str_in)

def add_zeros(imdb_list):
    new_imdb_list = []
    for item in imdb_list:
        a=str(item)
        #print len(a)
        if len(a)<7:
            item = (7-len(a))*'0'+a
        new_imdb_list.append(item)
    return new_imdb_list,str(len(new_imdb_list))

imdb_list = df.imdbId.unique()
new_imdb_list, number_of_movies = add_zeros(imdb_list)
print('the total number of movies is: '+number_of_movies+'\n')
df['content'] = '0'

content_dict = []
for item in new_imdb_list:
	#print item
	text = getcontect(item)
	ttext = unidecode(preprocess(text))
	#print ttext
	content_dict.append({'imdbID': item, 'content': ttext})

filename = '9066.csv'
pd.DataFrame.from_dict(content_dict).to_csv(filename, index=False, encoding='utf-8')
