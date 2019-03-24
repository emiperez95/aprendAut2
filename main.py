import numpy as np
from tree import tree

evaluationData = "data/evaluationData.npy"
trainingData = "data/trainingData.npy"
data = np.load(trainingData)

# Train models
treeObject = tree(data.astype(float))

# Evaluate models
