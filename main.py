import numpy as np
from tree import tree

evaluationData = "data/evaluationData.npy"
trainingData = "data/trainingData.npy"

fakeSet = np.array([
  ['1', '1'],
  ['2', '1'],
  ['3', '1'],
  ['0', '2'],

])
data = np.load(trainingData)

# Train models
treeObject = tree(data.astype(float))

# Evaluate models
