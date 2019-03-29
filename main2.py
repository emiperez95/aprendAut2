import numpy as np
from tree import makeNode
from poolTree import PoolTree
import random
import time
import pickle
import counter
from itertools import chain, combinations

def kFoldDataGen(data,k):
    dataLen = len(data)
    splitArr = [(i+1)*(dataLen//k) for i in range(k-1)]
    return np.split(data, splitArr)
 
# def evaluate(model, testData):
    


dataLoc = "dataSmall"

evaluationData = dataLoc+"/evaluationData.npy"
trainingData = dataLoc+"/trainingData.npy"
competitionData = dataLoc+"/competitionData.npy"
dataDump = "persist/temp2.th"
DUMP_SETTINGS = ("persist/", ".th") 

K_FOLD_PARTITIONS = 6

data = np.load(trainingData)
evData = np.load(competitionData)
lenEvData = len(evData)
kFoldData = kFoldDataGen(np.append(data, evData, 0), K_FOLD_PARTITIONS)

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

    # Train model
    # varA = True
    # varB = 2
    # model1 = makeNode(data, 54 , varA, varB)
    # print("Values: {}, {}".format(varA, varB))

    cla = 0
    attTypes = [0 for _ in range(10)] + [1 for _ in range(44)]
    attTypesTo2 = [1, 2, 3, 4, 6, 7, 8, 9]

    for att in attTypesTo2:
        attTypes[att] = 2

    start = time.time()

    print(len(arrTr))
    print(len(arrEv))

    model1 = makeNode(arrTr, 54, partitionStyle=False, entropyFunc=0, catTypeArr=attTypes)
    # model1 = PoolTree(data, 7 ,54, partitionStyle=False, entropyFunc=0, clasificador=1, attTypes=attTypes)

    middle = time.time()
    print("Training time: {}".format(middle-start))

    dataDump = DUMP_SETTINGS[0] + dataLoc + "_RUN_" + str(i) + DUMP_SETTINGS[1]
    with open(dataDump, 'wb') as f:
        pickle.dump(model1, f)

    # Evaluate models
    model1Score = 0

    for row in arrEv:
        res = model1.classify(row[:-1])

        if res == row[-1]:
            model1Score += 1

    print(" Modelo {}: {}".format(i, model1Score/lenEvData))



cla = 0
attTypes = [0 for _ in range(10)] + [1 for _ in range(44)]
attTypesTo2 = [1, 2, 3, 4, 6, 7, 8, 9]

for att in attTypesTo2:
    attTypes[att] = 2

start = time.time()

print(len(data))
print(len(evData))

model1 = makeNode(data, 54, partitionStyle=False, entropyFunc=0, catTypeArr=attTypes)
# model1 = PoolTree(data, 7 ,54, partitionStyle=False, entropyFunc=0, clasificador=1, attTypes=attTypes)

middle = time.time()
print("Training time: {}".format(middle-start))

dataDump = DUMP_SETTINGS[0] + dataLoc + "_RUN_" + str(i) + DUMP_SETTINGS[1]
with open(dataDump, 'wb') as f:
    pickle.dump(model1, f)

# Evaluate models
model1Score = 0

for row in evData:
    res = model1.classify(row[:-1])

    if res == row[-1]:
        model1Score += 1

print(" Modelo {}: {}".format(i, model1Score/lenEvData))
