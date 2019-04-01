import numpy as np
from tree import makeNode
from poolTree import PoolTree
import random
import time
import pickle
import counter
from itertools import chain, combinations
from evaluation import Evaluation


import sys
sys.setrecursionlimit(200000)

#=========Vars========
DATA_LOCATION = "data/"
DUMP_SETTINGS = ("data/", ".th") 
K_FOLD_PARTITIONS = 5
CLASS_AMM = 7

##==========AUX_FUNCS=======
def kFoldDataGen(data,k):
    dataLen = len(data)
    splitArr = [(i+1)*(dataLen//k) for i in range(k-1)]
    return np.split(data, splitArr)
# return spitData

def evaluate(model, testData):
    score = 0
    for row in testData:
        res = model.classify(row[:-1])
        if res == row[-1]:
            score += 1
    return score/len(testData)
# return score

def dumpModel(model, dir):
    with open(dir, 'wb') as f:
        pickle.dump(model, f)

def crossValidationTrain(kFold, data, classAmm, modelType, argv, dumpArgv):
    # Realiza la cross validation con k = kFOld, para la data = data,
        # y con el modelo modelType(0 = makeNode, 1 = nodeTree),
        # con los parametros argv pasados como array.
    # Retorna 3 arrays, los modelos, los tiempos de cada modelo y
        # los resultados de cada modelo

    kFoldData = kFoldDataGen(data, kFold)
    modelArr = []
    timeArr = []
    resultArr = []
    for i in range(len(kFoldData)):
        arrEv = []
        arrTr = []
        for j, arr in enumerate(kFoldData):
            if i == j:
                arrEv = arr
            else:
                if len(arrTr) == 0:
                    arrTr = arr
                else:
                    arrTr = np.append(arrTr, arr, 0)

        start = time.time()
        if modelType == 0:
            model = makeNode(*argv)
            argv[0] = arrTr
            dumpDir = dumpArgv[0] + "CV_" + str(i) + "_Func{}_Rep{}_Thr{}_Kfold{}_WHOLE".format(argv[3],argv[5], argv[6], kFold) + dumpArgv[1]
        else:
            model = PoolTree(*argv)
            argv[0] = arrTr
            dumpDir = dumpArgv[0] + "CV_" + str(i) + "_Func{}_Rep{}_Thr{}_Kfold{}_POOL".format(argv[4],argv[6], argv[7], kFold) + dumpArgv[1]
        modelArr.append(model)
        res = time.time()
        resTime = res-start
        print("Model time: ", resTime)
        model.setTime(resTime)
        timeArr.append(resTime)
        dumpModel(model, dumpDir)
        print("Dump time: ", time.time()-res)
        evalRes = Evaluation(model, arrEv, classAmm)
        resultArr.append(evalRes)
        print("______________")
        print(dumpDir)
        evalRes.normalPrint()
    return modelArr, timeArr, resultArr
    # return None, None, None
# return [model] [time] [score] 

def normalTrain(data, evData, classAmm, modelType, argv, dumpArgv):
    start = time.time()
    if modelType == 0:
        dumpDir = dumpArgv[0] + "NORMAL_" + "_Func{}_Rep{}_Thr{}_WHOLE".format(argv[3],argv[5], argv[6]) + dumpArgv[1]
        model = makeNode(*argv) 
    else: 
        dumpDir = dumpArgv[0] + "NORMAL_" + "_Func{}_Rep{}_Thr{}_POOL".format(argv[4],argv[6], argv[7]) + dumpArgv[1]
        model = PoolTree(*argv)
    timer = time.time()-start
    model.setTime(timer)
    dumpModel(model, dumpDir)
    score = Evaluation(model, evData, classAmm)
    print("______________")
    print(dumpDir)
    score.normalPrint()
    return model, timer, score
    # return None, None, None
# return mode, time, score

##=========MAIN CODE========

attTypes = [2 for _ in range(10)] + [1 for _ in range(44)]
for att in [0, 5]:
    attTypes[att] = 0        

#Feli
size = ["50k", "5k"]
repetition = [0,1]
funcs = [0,2]
threshs = [0.1, 0.01, 0.001]
folds = [5]

#Leo
size = ["500k"]
repetition = [1]
funcs = [0]
threshs = [0.01]
folds = [5]

#Yo
size = ["500k"]
repetition = [1]
funcs = [0]
threshs = [0.001]
folds = [5]

# size = ["500k", "50k", "5k"]
# repetition = [0,1]
# funcs = [0,2]
# threshs = [0.1, 0.01, 0.001]
# folds = [5]


trainingData = "/trainingData.npy"
evaluationData = DATA_LOCATION+"/evDataCover.npy"
competitionData = DATA_LOCATION+"/compDataCover.npy"
cmpData = np.load(competitionData)


for sz in size:
    print("=size",sz) #
    dumpArgv = [DUMP_SETTINGS[0]+ sz + "/", DUMP_SETTINGS[1]]
    trPath = DATA_LOCATION + sz + "/" + trainingData
    trData = np.load(trPath)
    for rep in repetition:
        print("==rep",rep) #
        if rep == 1:
            for f in funcs:
                if f == 2:
                    break

                print("===func", f) #
                for th in threshs:
                    print("====th", th) #
                    argvModel1 = [trData, 54, True, f, attTypes, rep, th]
                    argvModel2 = [trData, 7, 54, True, f, attTypes, rep, th]
                    for kFold in folds:
                        print("=====kFold", kFold) #
                        _, _, _ = crossValidationTrain(kFold, np.append(trData, cmpData, 0), CLASS_AMM, 0, argvModel1, dumpArgv)
                        # _, _, _ = crossValidationTrain(kFold, np.append(trData, cmpData), CLASS_AMM, 1, argvModel2, dumpArgv)
                    _, _, _ = normalTrain(trData, cmpData, CLASS_AMM, 0, argvModel1, dumpArgv)
                    _, _, _ = normalTrain(trData, cmpData, CLASS_AMM, 1, argvModel2, dumpArgv)

        else:
            for f in funcs:
                print("===func", f) #
                argvModel1 = [trData, 54, True, f, attTypes, rep, 0.0]
                argvModel2 = [trData, 7, 54, True, f, attTypes, rep, 0.0]
                for kFold in folds:
                    print("====kFold", kFold) #
                    _, _, _ = crossValidationTrain(kFold, np.append(trData, cmpData, 0), CLASS_AMM, 0, argvModel1, dumpArgv)
                    # _, _, _ = crossValidationTrain(kFold, np.append(trData, cmpData), CLASS_AMM, 1, argvModel2, dumpArgv)
                print("====noFold")
                _, _, _ = normalTrain(trData, cmpData, CLASS_AMM, 0, argvModel1, dumpArgv)
                _, _, _ = normalTrain(trData, cmpData, CLASS_AMM, 1, argvModel2, dumpArgv)
