import numpy as np
from tree import makeNode
from poolTree import PoolTree
from metrics import matrizConfusion
import random
import binarytree as bt
import time

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

evaluationData = "data/evaluationData.npy"
trainingData = "data/trainingData.npy"
competitionData = "data/competitionData.npy"

data = np.load(trainingData)
evData = np.load(competitionData)
lenEvData = len(evData)

data = data.astype(float)
evData = evData.astype(float)

start = time.time()

# Train models


classNameDict = {
    1: 'Iris Setosa',
    2: 'Iris Versicolour',
    3: 'Iris Virginica',
}

entropyFunc = [0, 1, 2]
partitionStyle = [True, False] #Model1
classifier = [0, 1] #Model2
for ent in entropyFunc:
    print()
    print("Entropy func: {} = ".format(ent), end="")
    for part, cla in zip(partitionStyle, classifier):
        model1 = makeNode(data, 4, part, ent, [0,0,0,0])
        model2 = PoolTree(data, 3, 4 , cla)

        # Evaluate models
        model1Score = 0
        model2Score = 0
        for row in evData:
            res = model1.classify(row[:-1])
            # print("  Respuesta modelo 1: ", res)
            if res == row[-1]:
                model1Score += 1

            res = model2.classify(row[:-1])
            # print("  Respuesta modelo 2: ", res)
            if res == row[-1]:
                model2Score += 1

        # print("PartitionStyle: {}, Classifier: {}".format(part, cla))
        # print("      Modelo 1: {}, part {}".format(model1Score/lenEvData, part))
        # print("      Modelo 2: {}, clas {}".format(model2Score/lenEvData, cla))
        print(" 1:{} - 2:{} =".format(model1Score/lenEvData, model2Score/lenEvData), end="")
        # print(nodeToBtNode(model1))
        # print(time.time()-start)
        # print()
