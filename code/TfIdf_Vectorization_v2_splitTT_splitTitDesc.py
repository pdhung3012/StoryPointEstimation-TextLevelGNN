from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk.tokenize import word_tokenize
import os
import numpy as np
import gensim
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.model_selection import train_test_split
import sys,os
sys.path.append(os.path.abspath(os.path.join('..')))
from UtilFunctions import createDirIfNotExist
import pandas as pd

fopRootData='../../../dataPapers/SEE/'
createDirIfNotExist(fopRootData)
fopVectorAllSystems=fopRootData+'result_tfidf/vec/'
fopVectorAllSystemsTrain=fopVectorAllSystems+'train/'
fopVectorAllSystemsTest=fopVectorAllSystems+'test/'
fopTextPreprocess=fopRootData+'result_tfidf/textPre/'
createDirIfNotExist(fopVectorAllSystems)
createDirIfNotExist(fopVectorAllSystemsTrain)
createDirIfNotExist(fopVectorAllSystemsTest)
createDirIfNotExist(fopTextPreprocess)

fopDataset='../dataset/'

# import stanza

def addDependenciesToSentence(docObj):
    lstSentences=docObj.sentences
    lstOutput=[]
    for sen in lstSentences:
        depends=sen._dependencies
        lstDepInfo=[]
        # depends=dict(depends)
        for deKey in depends:
            strElement=' '.join([deKey[2].text,deKey[0].text,deKey[1]])
            lstDepInfo.append(strElement)
        strDep=' '.join(lstDepInfo)
        lstOutput.append((strDep))
    strResult=' '.join(lstOutput)
    return strResult

def addDependenciesToSentenceCompact(docObj):
    lstSentences=docObj.sentences
    lstOutput=[]
    for sen in lstSentences:
        depends=sen._dependencies
        lstDepInfo=[]
        # depends=dict(depends)
        for deKey in depends:
            strElement=' '.join([deKey[1]])
            lstDepInfo.append(strElement)
        strDep=' '.join(lstDepInfo)
        lstOutput.append((strDep))
    strResult=' '.join(lstOutput)
    return strResult

def addDependenciesToSentencePOS(docObj):
    lstSentences=docObj.sentences
    lstOutput=[]
    for sen in lstSentences:
        words=sen._words
        lstDepInfo=[]
        # depends=dict(depends)
        for w in words:
            strElement=' '.join([w.upos])
            lstDepInfo.append(strElement)
        strDep=' '.join(lstDepInfo)
        lstOutput.append((strDep))
    strResult=' '.join(lstOutput)
    return strResult


def preprocess(textInLine):
    text = textInLine.lower()
    doc = word_tokenize(text)
    # doc = [word for word in doc if word in words]
    # doc = [word for word in doc if word.isalpha()]
    return ' '.join(doc)

from UtilFunctions import createDirIfNotExist



from os import listdir
from os.path import isfile, join
arrFiles = [f for f in listdir(fopDataset) if isfile(join(fopDataset, f))]
createDirIfNotExist(fopVectorAllSystems)
createDirIfNotExist(fopTextPreprocess)

# nlp = stanza.Pipeline() # This sets up a default neural pipeline in English

list_dir = os.listdir(fopDataset)   # Convert to lower case
list_dir =sorted(list_dir)
print(str(list_dir))
import time
start_time = time.time()

for filename in list_dir:
    if not filename.endswith('.csv'):
        continue
    #if not file.endswith('moodle.csv'):
    #    continue
    fileCsv = fopDataset + filename
   # fpVectorItemCate=fopVectorAllSystems+filename.replace('.csv','')+'_category.csv'
    fpVectorItemRegTrain = fopVectorAllSystemsTrain + filename
    fpVectorItemRegTest = fopVectorAllSystemsTest + filename
    nameProject=filename.replace('.csv','')
    df = pd.read_csv(fileCsv)
    df['title']=df['title'].values.astype('U')
    df['description'] = df['description'].values.astype('U')

    X_train, X_test, y_train, y_test = train_test_split(
        df[['title', 'description','storypoint','issuekey']],
        df['storypoint'], test_size=0.2, shuffle = False, stratify = None)

    # print(X_train['title'])
    vectorizer = TfidfVectorizer(ngram_range=(1, 1))
    title_vect_fit = vectorizer.fit(X_train['title'])
    tfidf_train_title = title_vect_fit.transform(X_train['title'])
    tfidf_test_title = title_vect_fit.transform(X_test['title'])

    vectorizer = TfidfVectorizer(ngram_range=(1, 1))
    description_vect_fit = vectorizer.fit(X_train['description'])
    tfidf_train_description = description_vect_fit.transform(X_train['description'])
    tfidf_test_description = description_vect_fit.transform(X_test['description'])

    X_train_vect = pd.concat([X_train[['storypoint','issuekey']].reset_index(drop=True),
                              pd.DataFrame(tfidf_train_title.toarray()),pd.DataFrame(tfidf_train_description.toarray())], axis=1)
    X_test_vect = pd.concat([X_test[['storypoint','issuekey']].reset_index(drop=True),
                              pd.DataFrame(tfidf_test_title.toarray()),pd.DataFrame(tfidf_test_description.toarray())], axis=1)
    X_train_vect.to_csv(fpVectorItemRegTrain, index=False)
    X_test_vect.to_csv(fpVectorItemRegTest, index=False)
    print('Finish {}'.format(filename))

end_time = time.time()
dur=end_time-start_time
print('Time duration: {}'.format(dur))
