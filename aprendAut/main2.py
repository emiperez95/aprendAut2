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


#Para elegir que casos correr, basta con modificar las variables anteriores.
    #Cabe notar que para tomar otros tamaños es necesario modificar el archivo dataPartitioner para que genere los datos necesarios.
    #Además, cuando se utiliza la función 2, no se usa crossvalitadion porque puede generar loops infinitos.
#size: corresponde a la cantidad de datos a entrenar.
    #Valores posibles: "500k", "50k","5k"
#Repetition: si es 1, los atributos pueden ser utilizados si superan el threshold. Si es 0, esto no sucede. 
    #Valores posibles: 0 o 1
#threshs: valores de thesholds para usar en caso que se use repetición.
    #Valores posibles: cualquier numero que pertenezca al intervalo [1.0, 0.0]
#funcs: 0 -> entropia de shanon, 1-> Gini, 2-> misclassification
    #Valores posibles: 0, 1 o 2
#folds: especfica la cantidad de folds a utilizar.
    #Valores posibles: cualquier número entero positivo
#===========
size = ["500k", "50k", "5k"]
repetition = [0,1]
funcs = [0,2]
threshs = [0.1, 0.01, 0.001]
folds = [5]

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

    funcDict = {
    0: "Entropia de Shanon",
    1: "Impureza de Gini",
    2: "Missclasification"
    }
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
            argv[0] = arrTr
            model = makeNode(*argv)
            dumpDir = dumpArgv[0] + "CV_" + str(i) + "_Func{}_Rep{}_Thr{}_Kfold{}_WHOLE".format(argv[3],argv[5], argv[6], kFold) + dumpArgv[1]
            printVar = "Corrida Cross Validation común: {}\nFunción: {}\nNúmero de folds: {}\nRepetición de atributos: {}\nThreshold de la repetición: {}".format(i+1, funcDict[argv[3]],kFold,argv[5], argv[6] )
        else:
            argv[0] = arrTr
            model = PoolTree(*argv)
            dumpDir = dumpArgv[0] + "CV_" + str(i) + "_Func{}_Rep{}_Thr{}_Kfold{}_POOL".format(argv[4],argv[6], argv[7], kFold) + dumpArgv[1]
            printVar = "Corrida Cross Validation con Bosque Binario: {}\nFunción: {}\nNúmero de folds: {}\nRepetición de atributos: {}\nThreshold de la repetición: {}".format(i+1, funcDict[argv[4]],kFold,argv[6], argv[7] )
        modelArr.append(model)
        res = time.time()
        resTime = res-start
        model.setTime(resTime)
        timeArr.append(resTime)
        dumpModel(model, dumpDir)
        evalRes = Evaluation(model, arrEv, classAmm)
        resultArr.append(evalRes)
        print("__________________________")
        print(printVar)
        print("Tiempo de ejeción: ", resTime)
        evalRes.normalPrint()
    return modelArr, timeArr, resultArr

def normalTrain(data, evData, classAmm, modelType, argv, dumpArgv):
    funcDict = {
    0: "Entropia de Shanon",
    1: "Impureza de Gini",
    2: "Missclasification"
    }
    start = time.time()
    if modelType == 0:
        dumpDir = dumpArgv[0] + "NORMAL_" + "_Func{}_Rep{}_Thr{}_WHOLE".format(argv[3],argv[5], argv[6]) + dumpArgv[1]
        model = makeNode(*argv)
        printVar = "Corrida común: \nFunción: {}\nRepetición de atributos: {}\nThreshold de la repetición: {}".format(funcDict[argv[3]],argv[5], argv[6] )
    else:
        dumpDir = dumpArgv[0] + "NORMAL_" + "_Func{}_Rep{}_Thr{}_POOL".format(argv[4],argv[6], argv[7]) + dumpArgv[1]
        printVar = "Corrida con Bosque Binario: \nFunción: {}\nRepetición de atributos: {}\nThreshold de la repetición: {}".format(funcDict[argv[4]],argv[6], argv[7] )
        model = PoolTree(*argv)
    timer = time.time()-start
    model.setTime(timer)
    dumpModel(model, dumpDir)
    score = Evaluation(model, evData, classAmm)
    print("______________")
    print(printVar)
    score.normalPrint()
    print("Tiempo de ejeción: ", timer)
    return model, timer, score

##=========MAIN CODE========

attTypes = [2 for _ in range(10)] + [1 for _ in range(44)]
for att in [0, 5]:
    attTypes[att] = 0

trainingData = "/trainingData.npy"
evaluationData = DATA_LOCATION+"/evDataCover.npy"
competitionData = DATA_LOCATION+"/compDataCover.npy"
cmpData = np.load(competitionData)

for sz in size:
    dumpArgv = [DUMP_SETTINGS[0]+ sz + "/", DUMP_SETTINGS[1]]
    trPath = DATA_LOCATION + sz + "/" + trainingData
    trData = np.load(trPath)
    for rep in repetition:
        if rep == 1:
            for f in funcs:
                if f == 2:
                    break

                for th in threshs:
                    argvModel1 = [trData, 54, True, f, attTypes, rep, th]
                    argvModel2 = [trData, 7, 54, True, f, attTypes, rep, th]
                    for kFold in folds:
                        _, _, _ = crossValidationTrain(kFold, np.append(trData, cmpData, 0), CLASS_AMM, 0, argvModel1, dumpArgv)
                        # _, _, _ = crossValidationTrain(kFold, np.append(trData, cmpData), CLASS_AMM, 1, argvModel2, dumpArgv)
                    _, _, _ = normalTrain(trData, cmpData, CLASS_AMM, 0, argvModel1, dumpArgv)
                    _, _, _ = normalTrain(trData, cmpData, CLASS_AMM, 1, argvModel2, dumpArgv)
        else:
            for f in funcs:
                argvModel1 = [trData, 54, True, f, attTypes, rep, 0.0]
                argvModel2 = [trData, 7, 54, True, f, attTypes, rep, 0.0]
                for kFold in folds:
                    _, _, _ = crossValidationTrain(kFold, np.append(trData, cmpData, 0), CLASS_AMM, 0, argvModel1, dumpArgv)
                    # _, _, _ = crossValidationTrain(kFold, np.append(trData, cmpData), CLASS_AMM, 1, argvModel2, dumpArgv)
                _, _, _ = normalTrain(trData, cmpData, CLASS_AMM, 0, argvModel1, dumpArgv)
                _, _, _ = normalTrain(trData, cmpData, CLASS_AMM, 1, argvModel2, dumpArgv)
