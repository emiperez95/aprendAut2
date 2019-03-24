import numpy as np
from tree import makeNode
from poolTree import PoolTree
import random

evaluationData = "data/evaluationData.npy"
trainingData = "data/trainingData.npy"
competitionData = "data/competitionData.npy"

data = np.load(trainingData)
evData = np.load(competitionData)
lenEvData = len(evData)

data = data.astype(float)
evData = evData.astype(float)

# Train models
model1 = makeNode(data)
model2 = PoolTree(data, 3)

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

print("Puntaje del modelo 1: ",model1Score/lenEvData)
print("Puntaje del modelo 2: ",model2Score/lenEvData)