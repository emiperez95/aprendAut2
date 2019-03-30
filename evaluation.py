import numpy as np
import matplotlib.pyplot as plt

class Evaluation:
  def __init__(self, model, testData, classAmm):
    self.classAmm = classAmm
    self.confussionMatrix = {
      i+1: { it+1:0 for it in range(classAmm) } for i in range(classAmm)
    }

    for elem in testData:
        res = model.classify(elem[:-1])
        self.confussionMatrix[elem[-1]][res] += 1
 
  def __str__(self):
    return str(self.confussionMatrix)

  def normalPrint(self):
    print('|-|', end="")
    for i in range(self.classAmm):
      print(i+1, " |", end="")
    print()
  
    print('|---:|', end="")
    for i in range(self.classAmm):
      print('---:|', end="")
    print()
  
    for row in self.confussionMatrix:
      print('|', row, ' |', end='')
      for col in self.confussionMatrix[row]:
        print(self.confussionMatrix[row][col], '|', end="")
      print()    

  def prettyPrintRes(self, classNameDict):
    print('|-|', end="")
    for i in range(len(classNameDict)):
      print(classNameDict[i+1], "|", end="")
    print()
  
    print('|---:|', end="")
    for i in range(len(classNameDict)):
      print('---:|', end="")
    print()
  
    for row in self.confussionMatrix:
      print('|', classNameDict[row], '|', end='')
      for col in self.confussionMatrix[row]:
        print(self.confussionMatrix[row][col], '|', end="")
      print()

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