import numpy as np
from tree import makeNode
from poolTree import PoolTree
import random
import binarytree as bt
import time
import pickle
from evaluation import Evaluation

# model1 = makeNode(data, 4, False, 2, [0,0,0,0])
# eval = Evaluation(model1, evData, 3)
# eval.normalPrint()
# eval.printMkdownStats()



#=========Vars========
DATA_LOCATION = "data"
DUMP_SETTINGS = ("persist/", ".th")
K_FOLD_PARTITIONS = 10
CLASS_AMM = 3

##==========AUX_FUNCS=======
def kFoldDataGen(data,k):
    dataLen = len(data)
    splitArr = [(i+1)*(dataLen//k) for i in range(k-1)]
    return np.split(data, splitArr)
# return spitData

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

def nodeToBtNode(nodo):
    if nodo.false_branch == None:
        btNode = bt.Node(nodo.mostCommonValue)
    else:
        btNode = bt.Node(nodo.cat)
        btNode.left = nodeToBtNode(nodo.true_branch)
        btNode.right = nodeToBtNode(nodo.false_branch)
    return btNode

def makeSimpleTree(metric):
    if metric == 'entropy':
        tree = makeNode(data, 4, False, 0, [0,0,0,0])
    elif metric == 'gini':
        tree = makeNode(data, 4, False, 1, [0,0,0,0])
    elif metric == 'misclassification':
        tree = makeNode(data, 4, False, 2, [0,0,0,0])
    print(tree)
    print(nodeToBtNode(tree))

evaluationData = DATA_LOCATION + "/evaluationData.npy"
trainingData = DATA_LOCATION + "/trainingData.npy"
competitionData = DATA_LOCATION + "/competitionData.npy"

data = np.load(trainingData)
evData = np.load(competitionData)
lenEvData = len(evData)
dumpArgv = [DUMP_SETTINGS[0] + DATA_LOCATION, DUMP_SETTINGS[1]]
data = data.astype(float)
evData = evData.astype(float)

compData = np.load(evaluationData).astype(float)

start = time.time()

# Train models


classNameDict = {
    1: 'Iris Setosa',
    2: 'Iris Versicolour',
    3: 'Iris Virginica',
}

attTypes = [0, 0, 0, 0]
argvModel1 = [data, 4, True, 0, attTypes, 0.01]
model1, timer, score = crossValidationTrain(K_FOLD_PARTITIONS, np.append(data, evData,0), CLASS_AMM, 0, argvModel1, dumpArgv)
print("| - | Micro Score | Macro Score |")
print("|--:|------------:|------------:|")
for ind, model in enumerate(model1):
    ev = Evaluation(model, evData, 3)
    print(nodeToBtNode(model))
    _, _, microFScore, _, _, macroFScore = ev.getStats()
    print('|',ind,'|', microFScore, '|', macroFScore, '|')
# model2 = PoolTree(data, 3, 4 , False, 2, [0,0,0,0])
# eval = Evaluation(model2, evData, 3)
# eval.prettyPrintRes(classNameDict)
# eval.printMkdownStats()

# entropyFunc = [0, 1, 2]
# partitionStyle = [True, False] #Model1
# classifier = [0, 1] #Model2
# for ent in entropyFunc:
#     print()
#     print("Entropy func: {} = ".format(ent), end="")
#     for part, cla in zip(partitionStyle, classifier):
#         model1 = makeNode(data, 4, part, ent, [0,0,0,0])
#         model2 = PoolTree(data, 3, 4 , cla)

#         # Evaluate models
#         model1Score = 0
#         model2Score = 0
#         for row in evData:
#             res = model1.classify(row[:-1])
#             # print("  Respuesta modelo 1: ", res)
#             if res == row[-1]:
#                 model1Score += 1

#             res = model2.classify(row[:-1])
#             # print("  Respuesta modelo 2: ", res)
#             if res == row[-1]:
#                 model2Score += 1

#         # print("PartitionStyle: {}, Classifier: {}".format(part, cla))
#         # print("      Modelo 1: {}, part {}".format(model1Score/lenEvData, part))
#         # print("      Modelo 2: {}, clas {}".format(model2Score/lenEvData, cla))
#         print(" 1:{} - 2:{} =".format(model1Score/lenEvData, model2Score/lenEvData), end="")
#         # print(nodeToBtNode(model1))
#         # print(time.time()-start)
#         # print()
