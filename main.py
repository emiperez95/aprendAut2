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

entropyFunc = [0, 1, 2]
partitionStyle = [True, False]
classifier = [0, 1]
for ent in entropyFunc:
    for part in partitionStyle:
        for cla in classifier:
            print("Entrpy func: {}, PartitionStyle: {}, Classifier: {}".format(ent, part, cla))
            
            model1 = makeNode(data, part, ent)
            model2 = PoolTree(data, 3, cla)

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

            print("      Modelo 1: ",model1Score/lenEvData)
            print("      Modelo 2: ",model2Score/lenEvData)
            print()
