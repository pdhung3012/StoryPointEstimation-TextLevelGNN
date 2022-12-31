# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 18:19:24 2019

@author: hungphd
"""
#nameSystem='titanium'


# import modules
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, train_test_split
import os
from sklearn.metrics import precision_score,accuracy_score
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error
import sys,os
sys.path.append(os.path.abspath(os.path.join('..')))
from UtilFunctions import createDirIfNotExist
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import glob


# Ppython program to check if two
# to get unique values from list
# using numpy.unique
import numpy as np


# function to get unique values
def unique(list1):
    x = np.array(list1)
    return np.unique(x)
def convertNormalLabelToTopLabel(originColumn):
    lstUnique=unique(originColumn)
    lstUnqSort=sorted(lstUnique)
    dictTop={}
    for i in range(1,len(lstUnqSort)+1):
        valInt=int(lstUnqSort[i-1])
        dictTop[valInt]=i
        dictReverse[i]=valInt
    lstNewColumn=[]
    for item in originColumn:
        newScore=dictTop[item]
        lstNewColumn.append(newScore)
    # print(dictReverse)
    return lstNewColumn

import math
def convertTopLabelToNormalLabel(topColumn):

    minValue=min(dictReverse.keys())
    maxValue=max(dictReverse.keys())
    lstNewColumn=[]
    for item in topColumn:
        rangeValue=0
        decVal, intVal = math.modf(item)
        intVal=int(intVal)
        if intVal <= minValue:
            intVal = 1
            rangeValue = dictReverse[minValue]
        elif intVal >= maxValue:
            rangeValue = 0
        else:
            rangeValue = dictReverse[intVal + 1] - dictReverse[intVal]
        if intVal == 1:
            realValue = dictReverse[intVal]
        else:
            realValue = dictReverse[intVal] + rangeValue * decVal
        lstNewColumn.append(realValue)
    return lstNewColumn

def scoreName(val):
    text='A'
    if val <= 5:
        text = 'A'
    elif val>5 and val<=15:
        text = 'B'
    elif val>15 and val<=40:
        text = 'C'
    else:
        text = 'D'
    return text

fopRootData='../../../dataPapers/SEE/'
createDirIfNotExist(fopRootData)
fopVectorAllSystems=fopRootData+'result_tfidf/vec/'
fopVectorAllSystemsTrain=fopVectorAllSystems+'train/'
fopVectorAllSystemsTest=fopVectorAllSystems+'test/'
fopOverallResultClass=fopRootData +'result_tfidf/class_reports/'

createDirIfNotExist(fopOverallResultClass)

dictReverse={}

from os import listdir
from os.path import isfile, join

fopOutputItemPredictedResult = fopOverallResultClass + "/results/"
createDirIfNotExist(fopOutputItemPredictedResult)
fpReportShort= fopOverallResultClass+'report_short.txt'
fpReportDetails= fopOverallResultClass+'report_details.txt'


lstFilePathProjects=sorted(glob.glob(fopVectorAllSystemsTrain+"*.csv"))

lstResultOverProjects=[]
lstStrResultOverProjects=[]

o2 = open(fpReportDetails, 'w')
o2.write('')
o2.close()

countA=0
countB=0
countC=0
countD=0

for file in lstFilePathProjects:
    if not file.endswith('.csv'):
        continue
    fileName=os.path.basename(file).replace('.csv', '')
    # fileCsv = fopVectorAllSystems + file+
    fpVectorItemClassTrain = fopVectorAllSystemsTrain + fileName + '.csv'
    fpVectorItemClassTest = fopVectorAllSystemsTest + fileName + '.csv'
    fpPredictedResult=fopOutputItemPredictedResult+fileName+'.txt'

    df_train = pd.read_csv(fpVectorItemClassTrain)
    # df_train[df_train['storypoint']<=5]='A'
    # df_train[df_train['storypoint'] >5 & df_train['storypoint'] <= 15] = 'B'
    # df_train[df_train['storypoint'] > 15 & df_train['storypoint'] <= 40] = 'C'
    # df_train[df_train['storypoint'] > 40 & df_train['storypoint'] <= 100] = 'D'
    df_train['storypoint'] = df_train['storypoint'].apply(scoreName)
    # print(list(df_train.columns.values))
    y_train = df_train['storypoint']
    X_train = df_train.drop(['storypoint','issuekey'],axis=1)

    dfTemp=df_train
    countA=countA+dfTemp[dfTemp['storypoint']=='A'].shape[0]
    countB = countB + dfTemp[dfTemp['storypoint'] == 'B'].shape[0]
    countC = countC + dfTemp[dfTemp['storypoint'] == 'C'].shape[0]
    countD = countD + dfTemp[dfTemp['storypoint'] == 'D'].shape[0]


    df_test = pd.read_csv(fpVectorItemClassTest)
    # df_test[df_test['storypoint'] <= 5] = 'A'
    # df_test[df_test['storypoint'] > 5 & df_test['storypoint'] <= 15] = 'B'
    # df_test[df_test['storypoint'] > 15 & df_test['storypoint'] <= 40] = 'C'
    # df_test[df_test['storypoint'] > 40 & df_test['storypoint'] <= 100] = 'D'
    df_test['storypoint'] = df_test['storypoint'].apply(scoreName)
    y_test = df_test['storypoint']
    X_test = df_test.drop(['storypoint','issuekey'], axis=1)

    dfTemp=df_test
    countA=countA+dfTemp[dfTemp['storypoint']=='A'].shape[0]
    countB = countB + dfTemp[dfTemp['storypoint'] == 'B'].shape[0]
    countC = countC + dfTemp[dfTemp['storypoint'] == 'C'].shape[0]
    countD = countD + dfTemp[dfTemp['storypoint'] == 'D'].shape[0]

print('{}\n{}\n{}\n{}'.format(countA,countB,countC,countD))

