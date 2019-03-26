import numpy as np
from tree import makeNode
from poolTree import PoolTree
import random


ogData = "data2/covtype.data"

trainingPath = "dataSuperSmall/trainingData.npy"
competitionPath = "dataSuperSmall/competitionData.npy"
evaluationPath = "dataSuperSmall/evaluationData.npy"

dataAmm = 5000
trainingAmm = round(dataAmm*0.6)+1
competitionAmm = evaluationAmm = round((dataAmm - trainingAmm)/2)

allData = []
with open(ogData) as fl:
    for f in fl:
        if not f == "":  
            row = f.split(",")
            row[-1] = row[-1][:-1]
            npRow = np.array(row)
            allData.append(npRow.astype(float))


print("Termine de leer")

random.shuffle(allData)

print("Termine el shuffle")

trainingData = []
competitionData = []
evaluationData = []

for i, row in enumerate(allData):
    if i < trainingAmm:
        trainingData.append(row)
    elif i < trainingAmm + competitionAmm:
        competitionData.append(row)
    else:
        evaluationData.append(row)

print("Particione la data")

np.save(trainingPath, trainingData)
np.save(competitionPath, competitionData)
np.save(evaluationPath, evaluationData)

print("Termine todo :) ")

# data = np.load(trainingData)
# evData = np.load(competitionData)
# lenEvData = len(evData)

# data = data.astype(float)
# evData = evData.astype(float)

# Train models