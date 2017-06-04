#coding:utf-8
import json
import pandas as pd
import os
import requests
import numpy as np
import re
from unidecode import unidecode
import sys
import codecs
#import xlrt,xlrd

df = pd.read_csv(os.getcwd()+'/df.csv', encoding='ISO-8859-1')
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

content_dict = []
sys.stdout = codecs.open('9066.csv', 'r+', 'utf-8')
sys.stdout.write('overview,imdbId\n')

for item in new_imdb_list:
    #print item
    try:
        text = getcontect(item)
        ttext = unidecode(preprocess(text))
        sys.stdout.write('%s,%s\n' % (ttext, item))
    except:
        sys.stdout.write(',%s\n' % item)
