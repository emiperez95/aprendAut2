import numpy as np
from tree import tree

evaluationData = "data/evaluationData.npy"
trainingData = "data/trainingData.npy"
data = np.load(trainingData)

# Train models
dic = {
    0:"Sepal length",
    1:"Selap width",
    2:"Petal length",
    3:"Petal width"
}
classDic = {
    1: "Iris setosa",
    2: "Iris versicolour",
    3: "Iris virginica"
}
treeObject = tree(data.astype(float), dic, classDic)

# Evaluate models
