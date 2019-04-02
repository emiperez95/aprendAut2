import numpy as np
from tree import makeNode
from poolTree import PoolTree
import random
import os

# +=======================CREACION DE DIRECTORIOS
print("Creando las carpetas", end='')
folders = ["500k", "50k", "5k"]
sizes = [500000, 50000, 5000]

for f in folders:
    path = "./data/"+f
    try:
        os.mkdir(path)
    except OSError:  
        print ("La creación del directorio %s fallo" % path)
    else:
        print('.', end='')

evPath = "data/evaDataCover.npy"
compPath = "data/compDataCover.npy"
trainPath = "/trainingData.npy"


# +======================= LECTURA DE DATOS
print("Leyendo datos...")
sourceData = "data/covtype.data"
allData = []
with open(sourceData) as fl:
    for f in fl:
        if not f == "":  
            row = f.split(",")
            row[-1] = row[-1][:-1]
            npRow = np.array(row)
            allData.append(npRow.astype(float))


# +======================= RANDOMIZACIÓN
print("Randomizando datos...")
random.shuffle(allData)


# =================Division de datos
print("Dividiendo los datos", end='')
fullSize = 581012
partitionSize = round(fullSize*0.2)

competitionData = []
evaluationData = []

for i in range(partitionSize):
    competitionData.append(allData[i*2])
    evaluationData.append(allData[(i*2)+1])
np.save(evPath, evaluationData)
np.save(compPath, competitionData)

print('.',end='')

for folder, size in zip(folders, sizes):

    trainingData = []
    minVal = min(fullSize, partitionSize*2+size)
    for i in range(partitionSize*2, minVal):
        trainingData.append(allData[i])
    np.save("./data/"+folder+trainPath, trainingData)
    print('.',end='')
print()
