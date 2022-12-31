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



fopRootData='../../../dataPapers/SEE/'
createDirIfNotExist(fopRootData)
fopVectorAllSystems=fopRootData+'result_tfidf/vec/'
fopVectorAllSystemsTrain=fopVectorAllSystems+'train/'
fopVectorAllSystemsTest=fopVectorAllSystems+'test/'
fopOverallResultReg=fopRootData +'result_tfidf/reg_reports/'

createDirIfNotExist(fopOverallResultReg)

dictReverse={}

from os import listdir
from os.path import isfile, join

fopOutputItemPredictedResult = fopOverallResultReg + "/results/"
createDirIfNotExist(fopOutputItemPredictedResult)
fpReportShort= fopOverallResultReg+'report_short.txt'
fpReportDetails= fopOverallResultReg+'report_details.txt'


lstFilePathProjects=sorted(glob.glob(fopVectorAllSystemsTrain+"*.csv"))

lstResultOverProjects=[]
lstStrResultOverProjects=[]

o2 = open(fpReportDetails, 'w')
o2.write('')
o2.close()

for file in lstFilePathProjects:
    if not file.endswith('.csv'):
        continue
    fileName=os.path.basename(file).replace('.csv', '')
    # fileCsv = fopVectorAllSystems + file+
    fpVectorItemRegTrain = fopVectorAllSystemsTrain + fileName + '.csv'
    fpVectorItemRegTest = fopVectorAllSystemsTest + fileName + '.csv'
    fpPredictedResult=fopOutputItemPredictedResult+fileName+'.txt'

    df_train = pd.read_csv(fpVectorItemRegTrain)
    print(list(df_train.columns.values))
    y_train = df_train['storypoint']
    X_train = df_train.drop(['storypoint','issuekey'],axis=1)

    df_test = pd.read_csv(fpVectorItemRegTest)
    y_test = df_test['storypoint']
    X_test = df_test.drop(['storypoint','issuekey'], axis=1)
    # print("********", "\n", "Random Forest Results Regression with: ", str(classifier))
    # X_train, X_test, y_train, y_test = train_test_split(all_data, all_label, test_size = 0.2,shuffle = False, stratify = None)
    # dictReverse={}
    classifier = RandomForestRegressor(n_estimators=100, max_depth=None, n_jobs=-1)
    classifier.fit(X_train, y_train)
    predicted = classifier.predict(X_test)

    #predicted=convertTopLabelToNormalLabel(predicted)
    maeAccuracy = mean_absolute_error(y_test, predicted)
    mqeAccuracy = mean_squared_error(y_test, predicted)

    print('{:.2f}'.format(maeAccuracy))

    np.savetxt(fpPredictedResult, predicted, fmt='%s', delimiter=',')
    o2 = open(fpReportDetails, 'a')
    o2.write(fileName+'\n')
    o2.write('Result for ' + str(classifier) + '\n')
    o2.write('MAE {}\nMQE {}\n\n\n'.format(maeAccuracy,mqeAccuracy))

    # o2.write(str(sum(cross_val) / float(len(cross_val))) + '\n')
    # o2.write(str(confusion_matrix(all_label, predicted)) + '\n')
    # o2.write(str(classification_report(all_label, predicted)) + '\n')
    o2.close()

    lstResultOverProjects.append(maeAccuracy)
    lstStrResultOverProjects.append('{}\t{}'.format(fileName,maeAccuracy))

averageAccuracy=np.average(lstResultOverProjects)
lstStrResultOverProjects.append('{}\t{}'.format('Avg',averageAccuracy))
f1=open(fpReportShort,'w')
f1.write('\n'.join(lstStrResultOverProjects))
f1.close()


