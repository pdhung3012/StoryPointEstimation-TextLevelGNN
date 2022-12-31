from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk.tokenize import word_tokenize
import os
import numpy as np
import gensim
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, train_test_split
import sys,os
print(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('..')))
from UtilFunctions import createDirIfNotExist
import torch

#handling text data
from torchtext.legacy import data
# from torchtext.legacy.data import *

import spacy
from spacy.lang.en import English
import networkx as nx
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from numpy import unique
import sys,os
sys.path.append(os.path.abspath(os.path.join('..')))
from UtilFunctions import createDirIfNotExist,scoreName
import nltk
nltk.download('punkt')

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

def putTextAndLabelToCSV(fpText,fpLabel,lstText,lstLbl,fpOut):
    f1=open(fpText,'r')
    arrText=f1.read().strip().split('\n')
    f1.close()
    f1 = open(fpLabel, 'r')
    arrLabel = f1.read().strip().split('\n')
    f1.close()

    lstStr = ['label,text']
    data=[]
    for i in range(0,len(arrText)):
        itemText=arrText[i].split('\t')[1].replace(',',' COMMA ')
        itemLabel=arrLabel[i].split('\t')[1].replace(',',' COMMA ')
        strItem='{},{}'.format(itemLabel,itemText)
        lstStr.append(strItem)
        tup=(itemText,itemLabel)
        data.append(tup)
        lstText.append(itemText)
        lstLbl.append(itemLabel)

    f1=open(fpOut,'w')
    f1.write('\n'.join(lstStr))
    f1.close()
    # df=pd.read_csv(fpOut,delimiter='\t')
    # print('lendf {}'.format(len(df)))
    return data


if __name__ == "__main__":

    fopFatherFolder = '../../../dataPapers/textInSPOC/mixCode_v2/step6/'

    fnSystemAbbrev = 'tlGNN/'
    fopOutputDs = fopFatherFolder+fnSystemAbbrev+'/'
    fpOutputTextIndex = fopFatherFolder+fnSystemAbbrev+'.txt'
    fpOutputTextTrainIndex = fopFatherFolder + fnSystemAbbrev + '.train.txt'
    fpOutputTestLbl= fopFatherFolder + fnSystemAbbrev + '_testLblStep1.txt'
    fpOutputPercentLbl = fopOutputDs + 'percentLabel.txt'
    fopOutputLabelInfo = fopOutputDs + 'labelInfo/'
    fopOutputLabelVocab = fopOutputDs + 'labelVocab/'
    fopRoot=fopFatherFolder

    createDirIfNotExist(fopOutputDs)
    createDirIfNotExist(fopOutputLabelInfo)
    createDirIfNotExist(fopOutputLabelVocab)

    fpTrain = fopRoot + 'train.csv'
    fpTextTrain = fopRoot + 'train.text.txt'
    fpLabelTrain = fopRoot + 'train.label.txt'

    fpValid = fopRoot + 'valid.csv'
    fpTextValid = fopRoot + 'testP.text.txt'
    fpLabelValid = fopRoot + 'testP.label.txt'

    fpTest = fopRoot + 'test.csv'
    fpTextTest = fopRoot + 'testW.text.txt'
    fpLabelTest = fopRoot + 'testW.label.txt'

    lstTextAll = []
    lstLabelAll=[]
    fpTextAll = fopRoot + 'text8'
    train_data = putTextAndLabelToCSV(fpTextTrain, fpLabelTrain, lstTextAll,lstLabelAll, fpTrain)
    valid_data = putTextAndLabelToCSV(fpTextValid, fpLabelValid, lstTextAll,lstLabelAll, fpValid)
    test_data = putTextAndLabelToCSV(fpTextTest, fpLabelTest, lstTextAll,lstLabelAll, fpTest)

    f1 = open(fpTextAll, 'w')
    f1.write('\n'.join(lstTextAll))
    f1.close()

    idx_train = len(train_data)
    idx_valid = len(train_data) + len(valid_data)
    idx_test = len(train_data) + len(valid_data) + len(test_data)
    all_data = train_data + valid_data + test_data

    # fnSystem=fnSystemAbbrev+'.csv'
    # fileCsv = fopDataset + fnSystem
    #
    # df = pd.read_csv(fileCsv)
    # columnId = df['issuekey']
    # df['storypoint'] = df['storypoint'].apply(scoreName)
    # columnRegStory = df['storypoint']
    # titles_and_descriptions = []
    # colTest=[]
    # for i in range(0, len(df['description'])):
    #     strContent = ' '.join([str(df['title'][i]), ' . ', str(df['description'][i])])
    #     strContent=preprocess(strContent).replace('\t',' ').replace('\n',' ').strip()
    #     # intValue=int(columnRegStory[i])
    #     # if(intValue>30):
    #     #     continue
    #
    #     titles_and_descriptions.append(str(strContent))
    #     colTest.append(str(columnRegStory[i]))
    TEXT = data.Field(tokenize='spacy', batch_first=True, include_lengths=True)
    LABEL = data.LabelField(dtype=torch.long, batch_first=True, use_vocab=True)
    fields = [('label', LABEL), ('text', TEXT)]
    # loading custom dataset
    # train_data = data.TabularDataset(path=fpTrain, format='csv', fields=fields, skip_header=True)
    # valid_data = data.TabularDataset(path=fpValid, format='csv', fields=fields, skip_header=True)
    # test_data = data.TabularDataset(path=fpTest, format='csv', fields=fields, skip_header=True)

    from torchtext.vocab import Vectors
    vectors = Vectors(name='vectors_glove.txt', cache=fopRoot)
    TEXT.build_vocab(train_data, valid_data, test_data, vectors=vectors)
    LABEL.build_vocab(train_data, valid_data, test_data)

    lstKeys=LABEL.vocab.itos
    dictLbl=LABEL.vocab.stoi

    dictTotalLabel = {}
    dictTotalStrContent = {}
    for i in range(0, len(lstLabelAll)):
        itemC = str(lstLabelAll[i])
        if itemC not in dictTotalLabel.keys():
            dictTotalLabel[itemC] = 1
            lstItem = []
            lstItem.append(lstTextAll[i])
            dictTotalStrContent[itemC] = lstItem
        else:
            dictTotalLabel[itemC] = dictTotalLabel[itemC] + 1
            lstItem = dictTotalStrContent[itemC]
            lstItem.append(lstTextAll[i])

    lstLogLabel = []
    for item in sorted(dictTotalLabel.keys()):
        val = dictTotalLabel[item]
        percent = (val * 1.0) / len(lstLabelAll)
        strLabel = '{}\t{}\t{}'.format(item, val, percent)
        lstLogLabel.append(strLabel)

        lstContentItem = dictTotalStrContent[item]
        fpLabelInfo = fopOutputLabelInfo + str(item) + '_info.txt'
        fff = open(fpLabelInfo, 'w')
        fff.write('\n'.join(lstContentItem))
        fff.close()

        dictItemFreq = {}
        for it2 in lstContentItem:
            arrTokens = word_tokenize(it2)
            for it in arrTokens:
                if it == '':
                    continue
                if not it in dictItemFreq.keys():
                    dictItemFreq[it] = 1
                else:
                    dictItemFreq[it] = dictItemFreq[it] + 1

        dictItemFreq = dict(sorted(dictItemFreq.items(), reverse=True, key=lambda item: item[1]))
        lstItemFreq = []
        for k in dictItemFreq.keys():
            strItem = '{}\t{}'.format(k, dictItemFreq[k])
            lstItemFreq.append(strItem)
        fpLabelFreq = fopOutputLabelVocab + str(item) + '_vocab.txt'
        fff = open(fpLabelFreq, 'w')
        fff.write('\n'.join(lstItemFreq))
        fff.close()

    # X_train_1, X_test, y_train_1, y_test = train_test_split(titles_and_descriptions, colTest, test_size=0.2, shuffle=False,
    #                                                     stratify=None)
    #
    # X_train, X_dev, y_train, y_dev = train_test_split(X_train_1, y_train_1, test_size=0.2,
    #                                                         shuffle=False,
    #                                                         stratify=None)
    '''                                                 
    X_train=X_train_1
    y_train=y_train_1
    X_dev=X_test
    y_dev=y_test
    '''

    #print('y test{}'.format(y_test))

    createDirIfNotExist(fopOutputDs)
    fpTextAll = fopOutputDs + fnSystemAbbrev + '-stemmed.txt'
    fpTextTrain=fopOutputDs+fnSystemAbbrev+'-train-stemmed.txt'
    fpTextDev = fopOutputDs + fnSystemAbbrev + '-dev-stemmed.txt'
    fpTextTest = fopOutputDs + fnSystemAbbrev + '-test-stemmed.txt'
    fpTextVocab = fopOutputDs + 'vocab.txt'
    fpTextVocab5 = fopOutputDs + 'vocab-5.txt'
    fpTextLabel = fopOutputDs + 'label.txt'
    fpTextFreq = fopOutputDs + 'freq.csv'

    lUniqueLabel=lstKeys
    fff=open(fpTextLabel,'w')
    fff.write('\n'.join(lUniqueLabel))
    fff.close()

    dictVocab={}
    dictVocab['UNK']=0
    for item in X_train_1:
        arrTokens=word_tokenize(item)
        for it in arrTokens:
            if it == '':
                continue
            if not it in dictVocab.keys():
                dictVocab[it]=1
            else:
                dictVocab[it]=dictVocab[it]+1

    listVoc=[]
    listFreq=[]

    for key in dictVocab.keys():
        listVoc.append(key)
        listFreq.append('{},{}'.format(key,dictVocab[key]))
    fff = open(fpTextVocab, 'w')
    fff.write('\n'.join(listVoc))
    fff.close()
    fff = open(fpTextVocab5, 'w')
    fff.write('\n'.join(listVoc))
    fff.close()
    fff = open(fpTextFreq, 'w')
    fff.write('\n'.join(listFreq))
    fff.close()

    listTDT = []
    for i in range(0, len(lstTextAll)):
        listTDT.append('{}\t{}'.format(lstLabelAll[i], lstTextAll[i]))
    fff = open(fpTextAll, 'w')
    fff.write('\n'.join(listTDT))
    fff.close()

    X_train=lstTextAll[:idx_train]
    y_train=lstLabelAll[:idx_train]
    X_valid=lstTextAll[idx_train:idx_valid]
    y_valid=lstLabelAll[idx_train:idx_valid]
    X_test=lstTextAll[idx_valid:idx_test]
    y_test=lstLabelAll[idx_valid:idx_test]
    listTDT=[]
    for i in range(0,len(X_train)):
        listTDT.append('{}\t{}'.format(y_train[i],X_train[i]))
    fff = open(fpTextTrain, 'w')
    fff.write('\n'.join(listTDT))
    fff.close()

    listTDT=[]
    for i in range(0,len(X_valid)):
        listTDT.append('{}\t{}'.format(y_valid[i],X_valid[i]))
    fff = open(fpTextDev, 'w')
    fff.write('\n'.join(listTDT))
    fff.close()

    listTDT=[]
    for i in range(0,len(X_test)):
        listTDT.append('{}\t{}'.format(y_test[i],X_test[i]))
    fff = open(fpTextTest, 'w')
    fff.write('\n'.join(listTDT))
    fff.close()


print('Done')




