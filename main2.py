import numpy as np
from tree import makeNode
from poolTree import PoolTree
import random
import time
import pickle
import counter
from itertools import chain, combinations
from evaluation import Evaluation

    

#=========Vars========
DATA_LOCATION = "dataSmall"
DUMP_SETTINGS = ("persist/", ".th") 
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
    print("Cross")

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
        else:
            model = PoolTree(*argv)
        modelArr.append(model)
        timeArr.append(time.time()-start)
        dumpDir = dumpArgv[0] + "_RUN_" + str(i) + "_{}_{}".format(argv[3],argv[5]) + dumpArgv[1]
        dumpModel(model, dumpDir)
        resultArr.append(Evaluation(model, arrEv, classAmm))
    return modelArr, timeArr, resultArr
# return [model] [time] [score] 

def normalTrain(data, evData, classAmm, modelType, argv, dumpArgv):
    start = time.time()
    model = makeNode(*argv) if modelType == 0 else PoolTree(*argv)
    dumpDir = dumpArgv[0] + "_RUN_" + "WHOLE" + "_{}_{}".format(argv[3],argv[5]) + dumpArgv[1]
    timer = time.time()-start
    dumpModel(model, dumpDir)
    score = Evaluation(model, evData, classAmm)
    return model, timer, score
# return mode, time, score

##=========MAIN CODE========
evaluationData = DATA_LOCATION+"/evaluationData.npy"
trainingData = DATA_LOCATION+"/trainingData.npy"
competitionData = DATA_LOCATION+"/competitionData.npy"
data = np.load(trainingData)
evData = np.load(competitionData)
dumpArgv = [DUMP_SETTINGS[0] + DATA_LOCATION, DUMP_SETTINGS[1]]

attTypes = [2 for _ in range(10)] + [1 for _ in range(44)]
for att in [0, 5]:
    attTypes[att] = 0
        

threshs = [0.2 , 0.1, 0.01, 0.001, 0.0001, 0.0]
folds = [5, 6, 7, 8, 9, 10]
funcs = []

for th in threshs:
    argvModel1 = [data, 54, True, 0, attTypes, th]
    print("Entre ", th)

    model1, timer, score = crossValidationTrain(K_FOLD_PARTITIONS, np.append(data, evData,0), CLASS_AMM, 0, argvModel1, dumpArgv)
    for tm, sc in zip(timer, score):
        print("Model time: {}".format(tm))
        sc.normalPrint()

    model2, timer2, score2 = normalTrain(data, evData, CLASS_AMM, 0, argvModel1, dumpArgv)
    print("Model time: {}".format(timer2))
    score2.normalPrint()
