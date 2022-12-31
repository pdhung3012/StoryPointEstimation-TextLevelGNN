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

'''
def getSentences(text):
    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    document = nlp(text)
    return [sent.string.strip() for sent in document.sents]
'''

def preprocess(textInLine):
    text = textInLine.lower()
    doc = word_tokenize(text)
    # doc = [word for word in doc if word in words]
    # doc = [word for word in doc if word.isalpha()]
    return ' '.join(doc)

from UtilFunctions import createDirIfNotExist

def printToken(token):
    print(token.text, "->", token.dep_)

def appendChunk(original, chunk):
    return original + ' ' + chunk

def isRelationCandidate(token):
    deps = ["ROOT", "adj", "attr", "agent", "amod"]
    return any(subs in token.dep_ for subs in deps)

def isConstructionCandidate(token):
    deps = ["compound", "prep", "conj", "mod"]
    return any(subs in token.dep_ for subs in deps)

def processSubjectObjectPairs(tokens):
    subject = ''
    object = ''
    relation = ''
    subjectConstruction = ''
    objectConstruction = ''
    for token in tokens:
#        printToken(token)
        if "punct" in token.dep_:
            continue
        if isRelationCandidate(token):
            relation = appendChunk(relation, token.lemma_)
        if isConstructionCandidate(token):
            if subjectConstruction:
                subjectConstruction = appendChunk(subjectConstruction, token.text)
            if objectConstruction:
                objectConstruction = appendChunk(objectConstruction, token.text)
        if "subj" in token.dep_:
            subject = appendChunk(subject, token.text)
            subject = appendChunk(subjectConstruction, subject)
            subjectConstruction = ''
        if "obj" in token.dep_:
            object = appendChunk(object, token.text)
            object = appendChunk(objectConstruction, object)
            objectConstruction = ''

    #print (subject.strip(), ",", relation.strip(), ",", object.strip())
    return (subject.strip(), relation.strip(), object.strip())

def processSentence(sentence,nlp_model):
    tokens = nlp_model(sentence)
    return processSubjectObjectPairs(tokens)

def returnIDOrGenNewID(content,dictVocab):
    result=-1
    if content in dictVocab.keys():
        result=dictVocab[content]
    else:
        result=len(dictVocab.keys())+1
        dictVocab[content]=result
    return result

def extractGraphKeyFromTriple(triples,dictVocab):
    G = nx.Graph()
    for triple in triples:
#        print('aaaa {}'.format(triple[0]))
        key0 = returnIDOrGenNewID(triple[0], dictVocab)
        key1 = returnIDOrGenNewID(triple[1], dictVocab)
        key2 = returnIDOrGenNewID(triple[2], dictVocab)
        G.add_node(key0)
        G.add_node(key1)
        G.add_node(key2)
        G.add_edge(key0, key1)
        G.add_edge(key1, key2)
    return G


def extractGraphSpekFromText(content,label,dictVocab,nlp_model,nlp):
    g=None
    sentences = getSentences(content, nlp)
    # nlp_model = spacy.load('en_core_web_sm')

    triples = []
    for sentence in sentences:
        triples.append(processSentence(sentence, nlp_model))
    g = extractGraphKeyFromTriple(triples, dictVocab)
    adjacency_matrix = nx.adjacency_matrix(g)
    graphSpek = Graph()
    graphSpek.a = adjacency_matrix
    graphSpek.y=label
    return graphSpek

if __name__ == "__main__":

    fopDataset = '../dataset/'
    fopFatherFolder = '../../../dataPapers/dataTextLevelPaper/'

    # fopRoot = '/home/hungphd/git/dataPapers/dataTextGCN/'

    list_dir = os.listdir(fopDataset)  # Convert to lower case
    list_dir = sorted(list_dir)

    for filename in list_dir:
        if not filename.endswith('.csv'):
            continue
        fnSystem = filename
        fnSystemAbbrev = filename.replace('.csv', '')
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

        fnSystem=fnSystemAbbrev+'.csv'
        fileCsv = fopDataset + fnSystem

        df = pd.read_csv(fileCsv)
        columnId = df['issuekey']
        df['storypoint'] = df['storypoint'].apply(scoreName)
        columnRegStory = df['storypoint']
        titles_and_descriptions = []
        colTest=[]
        for i in range(0, len(df['description'])):
            strContent = ' '.join([str(df['title'][i]), ' . ', str(df['description'][i])])
            strContent=preprocess(strContent).replace('\t',' ').replace('\n',' ').strip()
            # intValue=int(columnRegStory[i])
            # if(intValue>30):
            #     continue

            titles_and_descriptions.append(str(strContent))
            colTest.append(str(columnRegStory[i]))

        dictTotalLabel = {}
        dictTotalStrContent = {}
        for i in range(0, len(colTest)):
            itemC = str(colTest[i])
            if itemC not in dictTotalLabel.keys():
                dictTotalLabel[itemC] = 1
                lstItem = []
                lstItem.append(titles_and_descriptions[i])
                dictTotalStrContent[itemC] = lstItem
            else:
                dictTotalLabel[itemC] = dictTotalLabel[itemC] + 1
                lstItem = dictTotalStrContent[itemC]
                lstItem.append(titles_and_descriptions[i])

        lstLogLabel = []
        for item in sorted(dictTotalLabel.keys()):
            val = dictTotalLabel[item]
            percent = (val * 1.0) / len(colTest)
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

        X_train_1, X_test, y_train_1, y_test = train_test_split(titles_and_descriptions, colTest, test_size=0.2, shuffle=False,
                                                            stratify=None)

        X_train, X_dev, y_train, y_dev = train_test_split(X_train_1, y_train_1, test_size=0.2,
                                                                shuffle=False,
                                                                stratify=None)
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

        lUniqueLabel=unique(colTest)
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
        for i in range(0, len(titles_and_descriptions)):
            listTDT.append('{}\t{}'.format(colTest[i], titles_and_descriptions[i]))
        fff = open(fpTextAll, 'w')
        fff.write('\n'.join(listTDT))
        fff.close()

        listTDT=[]
        for i in range(0,len(X_train)):
            listTDT.append('{}\t{}'.format(y_train[i],X_train[i]))
        fff = open(fpTextTrain, 'w')
        fff.write('\n'.join(listTDT))
        fff.close()

        listTDT=[]
        for i in range(0,len(X_dev)):
            listTDT.append('{}\t{}'.format(y_dev[i],X_dev[i]))
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




