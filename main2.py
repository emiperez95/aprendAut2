import numpy as np
from tree import makeNode
from poolTree import PoolTree
import random
import time
import pickle
import counter

start = time.time()

evaluationData = "dataSuperSmall/evaluationData.npy"
trainingData = "dataSuperSmall/trainingData.npy"
competitionData = "dataSuperSmall/competitionData.npy"

dataDump = "persist/temp2.th"

data = np.load(trainingData)
evData = np.load(competitionData)
lenEvData = len(evData)

# Train model
# varA = True
# varB = 2
# model1 = makeNode(data, 54 , varA, varB)
# print("Values: {}, {}".format(varA, varB))

cla = 0
# model1 = PoolTree(data, 7, 54, cla)
attTypes = [0 for _ in range(10)] + [1 for _ in range(44)]
attTypesTo2 = [8]
for att in attTypesTo2:
    attTypes[att] = 2
# cnt = counter.Counter()

model1 = makeNode(data, 54, False, 0, attTypes)

middle = time.time()
print("Training time: {}".format(middle-start))

with open(dataDump, 'wb') as f:
    pickle.dump(model1, f)
dumpTime = time.time()
print("Dump time: {}".format(dumpTime-middle))

# Evaluate models
model1Score = 0

for row in evData:
    res = model1.classify(row[:-1])

    if res == row[-1]:
        model1Score += 1

end = time.time()
print("Final time: {}".format(end-dumpTime))

# print(model1)

print(" Modelo 1: {}".format(model1Score/lenEvData))
