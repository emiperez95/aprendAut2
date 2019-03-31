import numpy as np
import matplotlib.pyplot as plt

class Evaluation:
  def __init__(self, model, testData, classAmm):
    self.classAmm = classAmm
    self.confussionMatrix = {
      i+1: { it+1:0 for it in range(classAmm) } for i in range(classAmm)
    }
    totalScored = 0
    for elem in testData:
        res = model.classify(elem[:-1])
        self.confussionMatrix[elem[-1]][res] += 1
        if res == elem[-1]:
          totalScored += 1
    self.totalPrecisionPercentage = totalScored*100/len(testData)

  def __str__(self):
    return str(self.confussionMatrix)

  def __getRecArr(self):
    resArr = []
    for i in range(self.classAmm):
      row = self.confussionMatrix[i+1]
      truePos = 0
      falseNeg = 0
      for j in range(self.classAmm):
        if i == j:
          truePos = row[j+1]
        else:
          falseNeg += row[j+1]
      resArr.append((truePos, falseNeg))
    return resArr
    
  def __getPrecArr(self):
    resArr = []
    for i in range(self.classAmm):
      truePos = 0
      falsePos = 0
      for j in range(self.classAmm):
        if i == j:
          truePos = self.confussionMatrix[j+1][i+1]
        else:
          falsePos += self.confussionMatrix[j+1][i+1]
      resArr.append((truePos, falsePos))
    return resArr

  def getStats(self):
    precArr = self.__getPrecArr()
    recArr = self.__getRecArr()
    # Micro
    truePosSum = 0
    falsePosSum = 0
    falseNegSum = 0

    for i, elem in enumerate(precArr):
      truePosSum += elem[0]
      falsePosSum += elem[1]
      falseNegSum += recArr[i][1]
    microPrecision = truePosSum / (truePosSum+falsePosSum)
    microRecal = truePosSum / (truePosSum + falseNegSum)
    microFscore = 2*microPrecision*microRecal / (microPrecision + microRecal)

    # Macro
    precResSum = 0
    recResSum = 0
    for i, elem in enumerate(precArr):
      # self.confMatrix()
      # input("")
      if (elem[0] + elem[1]) == 0:
        precResSum += 0
      else:
        precRes = elem[0] / (elem[0] + elem[1])
        precResSum += precRes
      recRes = recArr[i][0] / (recArr[i][0] + recArr[i][1])
      recResSum += recRes
    macroPrecision = precResSum / self.classAmm
    macroRecal = recResSum / self.classAmm
    macroFscore = 2*macroPrecision*macroRecal / (macroPrecision + macroRecal)

    return microPrecision, microRecal, microFscore, macroPrecision, macroRecal, macroFscore
    
  def printStats(self):
    microPrecision, microRecal, microFscore, macroPrecision, macroRecal, macroFscore = self.getStats()
    table_data = [
      ["", "Prec", "Rec", "Fs"],
      ["Micro", round(microPrecision, 3), round(microRecal, 3), round(microFscore, 3)],
      ["Macro", round(macroPrecision, 3), round(macroRecal, 3), round(macroFscore, 3)]
    ]
    for row in table_data:
      print("{: >7} {: >8} {: >8} {: >8}".format(*row))

  def printMkdownStats(self):
      microPrecision, microRecal, microFscore, macroPrecision, macroRecal, macroFscore = self.getStats()
      table_data = [
        ["|-", "|Prec", "|Rec", "|Fs |"],
        ["|---:","|---:","|---:","|---:|"],
        ["|Micro", "|"+str(round(microPrecision, 3)), "|"+str(round(microRecal, 3)), "|"+str(round(microFscore, 3))+"|"],
        ["|Macro", "|"+str(round(macroPrecision, 3)), "|"+str(round(macroRecal, 3)), "|"+str(round(macroFscore, 3))+"|"]
      ]
      for row in table_data:
        print("{} {} {} {}".format(*row))

  def normalPrint(self):
    print("Total precision percentage: ", self.totalPrecisionPercentage)
    print()
    self.printStats()
    print()
    self.confMatrix()

  def confMatrix(self):
    print("Confussion Matrix: ")
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
