import numpy as np
from tree import tree
from poolTree import PoolTree
import random

evaluationData = "data/evaluationData.npy"
trainingData = "data/trainingData.npy"
competitionData = "data/competitionData.npy"

ev1Data = "data/evData1.npy"
ev2Data = "data/evData2.npy"
ev3Data = "data/evData3.npy"

data = np.load(trainingData)

# # Train models
# dic = {
#     0:"Sepal length",
#     1:"Selap width",
#     2:"Petal length",
#     3:"Petal width"
# }
# classDic = {
#     1: "Iris setosa",
#     2: "Iris versicolour",
#     3: "Iris virginica"
# }
# treeObject = tree(data.astype(float))


pTree = PoolTree(data.astype(float), 3)

checkTuple = [5.4, 3.4, 1.7, 0.2]
res = pTree.classify(checkTuple)
print(res)
# Evaluate models

