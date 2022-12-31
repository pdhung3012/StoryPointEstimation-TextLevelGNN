from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk.tokenize import word_tokenize
import os
import numpy as np
import gensim
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, train_test_split



import spacy
from spacy.lang.en import English
import networkx as nx
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from numpy import unique
import sys,os
sys.path.append(os.path.abspath(os.path.join('..')))
from UtilFunctions import createDirIfNotExist,scoreName,preprocessFollowingNLPStandard
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')

ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')


def initDefaultTextEnvi():
    nlp_model = spacy.load('en_core_web_sm')
    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    return nlp_model,nlp

def getSentences(text,nlp):
    result=None
    try:
        document = nlp(text)
        result= [sent.string.strip() for sent in document.sents]
    except Exception as e:
        print('sone error occured {}'.format(str(e)))
    return result

def preprocess(textInLine):
    text = textInLine.lower()
    doc = word_tokenize(text)
    # doc = [word for word in doc if word in words]
    # doc = [word for word in doc if word.isalpha()]
    return ' '.join(doc)


import codecs
if __name__ == "__main__":

    fopDataset = '../dataset/'
    fopFatherFolder = '../../../dataPapers/POSTagDataset/pythondocstring_ds/'
    fpInputText='../../../dataPapers/POSTagDataset/code_docstring_input/data_ps.descriptions.train'
    createDirIfNotExist(fopFatherFolder)



    # fopRoot = '/home/hungphd/git/dataPapers/dataTextGCN/'

    list_dir = os.listdir(fopDataset)  # Convert to lower case
    list_dir = sorted(list_dir)

    indexLineWrite=0
    indexFileWrite = 1
    maxLineWrite=10000

    fnRawText='rawText_part{}.txt'.format(indexFileWrite)
    fnPreprocessText = 'preprocessText_part{}.txt'.format(indexFileWrite)
    fnPOSTag = 'pos_part{}.txt'.format(indexFileWrite)
    fnTime = 'time{}.txt'.format(indexFileWrite)

    fff=codecs.open(fpInputText, 'r',  encoding="latin-1")
    arrContent=fff.read().split('\n')
    fff.close()


    lstText=[]
    lstPre=[]
    lstPOS=[]
    lstTime=[]

    for i in range(0, len(arrContent)):
        strTitle = arrContent[i]
        isRunOK=False
        try:
            strText,strPre,strPOS,run_time=preprocessFollowingNLPStandard(strTitle,ps,lemmatizer)
            isRunOK=True
        # print(strPre)
        except:
            isRunOK=False

        if isRunOK:
            indexLineWrite=indexLineWrite+1
            lstText.append(strText)
            lstPre.append(strPre)
            lstPOS.append(strPOS)
            lstTime.append(str(run_time))
        if((indexLineWrite % maxLineWrite)==0):
            print('go here')
            indexLineWrite=0
            indexFileWrite=indexFileWrite+1
            fff=open(fopFatherFolder+fnRawText,'w')
            fff.write('\n'.join(lstText))
            fff.close()
            fff = open(fopFatherFolder + fnPreprocessText, 'w')
            fff.write('\n'.join(lstPre))
            fff.close()
            fff = open(fopFatherFolder + fnPOSTag, 'w')
            fff.write('\n'.join(lstPOS))
            fff.close()
            fff=open(fopFatherFolder+fnTime,'w')
            fff.write('\n'.join(lstTime))
            fff.close()

            fnRawText = 'rawText_part{}.txt'.format(indexFileWrite)
            fnPreprocessText = 'preprocessText_part{}.txt'.format(indexFileWrite)
            fnPOSTag = 'pos_part{}.txt'.format(indexFileWrite)
            fnTime = 'time{}.txt'.format(indexFileWrite)
            print('start write {}'.format(fnRawText))

    if len(lstText)>0:
        indexFileWrite = indexFileWrite + 1
        fff = open(fopFatherFolder + fnRawText, 'w')
        fff.write('\n'.join(lstText))
        fff.close()
        fff = open(fopFatherFolder + fnPreprocessText, 'w')
        fff.write('\n'.join(lstPre))
        fff.close()
        fff = open(fopFatherFolder + fnPOSTag, 'w')
        fff.write('\n'.join(lstPOS))
        fff.close()
        fff = open(fopFatherFolder + fnTime, 'w')
        fff.write('\n'.join(lstTime))
        fff.close()
        lstText = []
        lstPOS = []
        lstPre = []
        lstTime = []

        fnRawText = 'rawText_part{}.txt'.format(indexFileWrite)
        fnPreprocessText = 'preprocessText_part{}.txt'.format(indexFileWrite)
        fnPOSTag = 'pos_part{}.txt'.format(indexFileWrite)
        fnTime = 'time{}.txt'.format(indexFileWrite)




print('Done')




