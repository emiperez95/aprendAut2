
import numpy as np
import matplotlib.pyplot as plt


def matrizConfusion(evData, classNameDict, model):
  values = evData[:,-1].astype(np.int)
  countPerValue = np.bincount(values)
  existingClasses = np.nonzero(countPerValue)[0]
  countPerValue = countPerValue[existingClasses]
  # existingClasses = map(lambda x: classNameDict[x], existingClasses)

  accuracyCounter = [0 for i in range(1, len(classNameDict)+1)]

  confussionMatrix = {
    i+1: { it+1:0 for it in range(len(classNameDict)) } for i in range(len(classNameDict))
  }

  # Evaluate model
  for elem in evData:
    res = model.classify(elem[:-1])
    confussionMatrix[elem[-1]][res] += 1
    # if res == elem[-1]:
    #   accuracyCounter[int(elem[-1]-1)] += 1

  print(confussionMatrix)
  print('|-|', end="")
  for i in range(len(classNameDict)):
    print(classNameDict[i+1], "|", end="")
  print()

  print('|---:|', end="")
  for i in range(len(classNameDict)):
    print('---:|', end="")
  print()

  for row in confussionMatrix:
    print('|', classNameDict[row], '|', end='')
    for col in confussionMatrix[row]:
      print(confussionMatrix[row][col], '|', end="")
    print()

  # accuracyPercentages = {}
  # for ind, res in enumerate(accuracyCounter):
  #   accuracyPercentages[ind+1] = res*100/countPerValue[ind]

  # print('|Class|Count|Count Predicted|Percentage Predicted|')
  # print('|----:|----:|--------------:|-------------------:|')
  # for i, c in enumerate(accuracyCounter):
  #   print('|', classNameDict[i+1], '|', countPerValue[i], '|', c, '|', accuracyPercentages[i+1], '%', '|')

  # print('Cleaning: ...')
  # print(cleanTree(model))

def isLeaf(root):
  return not root.false_branch and not root.true_branch

def MCV(root):
  return root.mostCommonValue

def cleanTree(root):
  if (isLeaf(root)):
    return root
  root.false_branch = cleanTree(root.false_branch)
  root.true_branch = cleanTree(root.true_branch)

  if (isLeaf(root.false_branch) and isLeaf(root.true_branch)):
    if (
      root.percentage == root.false_branch.percentage == root.true_branch.percentage
      and MCV(root) == MCV(root.false_branch) == MCV(root.true_branch)
      ):
      root.false_branch = None
      root.true_branch = None
  return root
