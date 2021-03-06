import numpy as np
from functools import reduce
from node import Node
import threading
import time as tm
import counter


def makeNode (trainingData, catAmm, partitionStyle = True, entropyFunc = 0, catTypeArr = None, repetition = 0, entThresh=0.0, catDict = None, classDict = None, counter = None):
  # Partition style: "False" for partition with <, "True" for <= (see __partition() )
  # entropyFunc: 0 -> shannonEntropy, 1 -> giniImpuruty, 2 -> misclassification.
  if catTypeArr == None:
    catTypeArr = [0 for _ in range(catAmm)]

  varDict = {
    "entropy" : float(__entropy(trainingData, entropyFunc)),
    "lengthTS" : float(len(trainingData)),
    "catDict" : catDict,
    "classDict" : classDict,
    "partitionStyle" : partitionStyle,
    "entropyFunc" : entropyFunc,
    "catTypeArr" : catTypeArr,
    "counter" : counter,
    "entThresh": entThresh,
    "repetition" : repetition
  }
  nodo = Node(varDict["catDict"], varDict["classDict"])
  __id3(nodo, varDict, trainingData, catAmm, [i for i in range(catAmm)])
  return nodo

def __entropy(examples, entropyFunc):
  if entropyFunc  == 0:
    return __shannonEntropy(examples)
  elif entropyFunc == 1:
    return __giniImpurity(examples)
  elif entropyFunc == 2:
    return __misclassification(examples)
  else:
    return __shannonEntropy(examples)

def __shannonEntropy(examples):
  entropy = 0
  # Si abajo en Pi = .. uno de los dos no es float, da 0 como resultado por ser ints
  length = float(len(examples))
  counts = __countOfEachClass(examples)
  for count in counts:
    Pi = count[1]/length
    entropy -= Pi* np.log2(Pi)
  return entropy

def __giniImpurity(examples):
  impurity = 0
  length = float(len(examples))
  counts = __countOfEachClass(examples)
  for count in counts:
    Pi = count[1]/length
    impurity += Pi*(1 - Pi)
  return impurity

def __misclassification(examples):
  length = float(len(examples))
  counts = __countOfEachClass(examples)
  maxCount = max(counts, key=lambda item: item[1])[1] if len(examples) != 0 else 0
  return (1 - (maxCount/length)) if length else 1

def __id3(newRootNode, varDict, examples, target_attribute, attributes):
  countOfEachClass = __countOfEachClass(examples)
  mostCommonValue, percentage = __mostCommonValue(countOfEachClass)
  # Asigno porcentaje y valor mas comun a todos los nodos
  newRootNode.percentage = percentage
  newRootNode.mostCommonValue = mostCommonValue
  newRootNode.countOfEach = countOfEachClass
  if (len(countOfEachClass) == 1):
    newRootNode.nodeClass = countOfEachClass[0][0]
    return newRootNode
  if (len(attributes) == 0):
    newRootNode.nodeClass = mostCommonValue
    return newRootNode
  bestAttribute, threshold, partitionLess, partitionEqualGreat, bestIG = __bestFitAttribute(varDict, examples, attributes)
  if varDict["catTypeArr"][bestAttribute]== 1 or varDict["repetition"] == 0:
    sonAttr = list(set(attributes) - {bestAttribute})
  else:
    sonAttr = attributes
  newRootNode.cat = bestAttribute
  newRootNode.threshold = threshold
  newRootNode.gain = bestIG
  newRootNode.nodeClass = mostCommonValue

  if varDict["repetition"] == 1:
    if bestIG <= varDict["entThresh"]:
      return newRootNode

  if (len(partitionLess) == 0):
    newRootNode.false_branch = Node(varDict["catDict"], varDict["classDict"])
    newRootNode.false_branch.nodeClass = mostCommonValue
    newRootNode.false_branch.mostCommonValue = mostCommonValue
    newRootNode.false_branch.percentage = percentage
    newRootNode.false_branch.countOfEach = []
  else:
    newRootNode.false_branch = Node(varDict["catDict"], varDict["classDict"])
    __id3(newRootNode.false_branch,varDict, partitionLess, target_attribute, sonAttr)
  if (len(partitionEqualGreat) == 0):
    newRootNode.true_branch = Node(varDict["catDict"], varDict["classDict"])
    newRootNode.true_branch.nodeClass = mostCommonValue
    newRootNode.true_branch.mostCommonValue = mostCommonValue
    newRootNode.true_branch.percentage = percentage
    newRootNode.true_branch.countOfEach = []
  else:
    newRootNode.true_branch = Node(varDict["catDict"], varDict["classDict"])
    __id3(newRootNode.true_branch, varDict, partitionEqualGreat, target_attribute, sonAttr)

def __mostCommonValue(countOfEachClass):
  totalCount = 0
  for elem in countOfEachClass:
    totalCount += elem[1]
  maxCount = max(countOfEachClass, key=lambda item: item[1])
  return maxCount[0], maxCount[1]*100/totalCount

def __bestFitAttribute(varDict, examples, attributes):
  # Si entra aca es porque hay attributes y al menos 2 ejemplos con distinta clase
  # Sino hubiese entrado en un caso base.
  bestIG = None
  bestThreshold = None
  bestAttribute = None
  bestPartitionLess = None
  bestPartitionEqualGreat = None

  for attr in attributes:
    examplesEntropy = __entropy(examples, varDict["entropyFunc"])
    IG, threshold, partitionLess, partitionEqualGreat = __gainAndThreshold(varDict, examples, attr, examplesEntropy)
    if bestIG == None or bestIG < IG:
      bestIG = IG
      bestThreshold = threshold
      bestAttribute = attr
      bestPartitionLess = partitionLess
      bestPartitionEqualGreat = partitionEqualGreat

  return bestAttribute, bestThreshold, bestPartitionLess, bestPartitionEqualGreat, bestIG

def __gainAndThreshold(varDict, examples, attribute, examplesEntropy):
  # Lets get the possible thresholds
  sortedExamples = examples[np.argsort(examples[:,attribute])]
  possibleThresholds = []
  begin = tm.time()

  catTypeArr = varDict["catTypeArr"]
  if catTypeArr[attribute] == 1:
    possibleThresholds.append(0.5)
  else:
    lastClass = sortedExamples[0][-1]
    lastAttrValue = sortedExamples[0][attribute]
    for row in sortedExamples[1:]:
      if row[-1] != lastClass:
        possibleThresholds.append((row[attribute]+lastAttrValue)/2)
      lastClass = row[-1]
      lastAttrValue = row[attribute]

  # Lets get the best threshold
  bestThreshold = None
  bestIG = None
  bestPartitionLess = None
  bestPartitionEqualGreat = None

  # Remover duplicados de possibleThresholds
  possibleThresholds = list(set(possibleThresholds))
  for threshold in possibleThresholds:
    dividerRow = np.argmax(sortedExamples[:,attribute] > threshold)
    lessLength = 0
    equalGreatLength = len(examples) - dividerRow
    if (dividerRow > 0 or (dividerRow == 0 and examples[0][attribute] <= threshold)):
      lessLength = dividerRow
    partitionLess = sortedExamples[:lessLength]
    partitionEqualGreat = sortedExamples[dividerRow:]
    if (len(partitionLess) == 0 or len(partitionEqualGreat) == 0) and varDict["entropyFunc"]!=2 and False:
      IG = 0.0
    else:
      HLess = (lessLength/len(examples))*__entropy(partitionLess, varDict["entropyFunc"])
      HEqualGreat = (equalGreatLength/len(examples))*__entropy(partitionEqualGreat, varDict["entropyFunc"])
      IG = examplesEntropy - HLess - HEqualGreat

      if catTypeArr[attribute] == 2 and IG != 0:
        if len(partitionLess) == 0:
          lesser = 0
        else:
          lesser = ( len(partitionLess) / len(examples) ) * np.log( len(partitionLess) / len(examples) )
        if len(partitionEqualGreat) == 0:
          greater = 0
        else:
          greater = (len(partitionEqualGreat)/len(examples))*np.log(len(partitionEqualGreat)/len(examples))
        splitInfo = -lesser - greater

        if splitInfo != 0:
          IG = IG / splitInfo

    if (bestIG == None or bestIG < IG):
      bestThreshold = threshold
      bestIG = IG
      bestPartitionLess = partitionLess
      bestPartitionEqualGreat = partitionEqualGreat
  return bestIG, bestThreshold, bestPartitionLess, bestPartitionEqualGreat

def __countOfEachClass(examples):
  values = examples[:,-1].astype(np.int)
  countPerValue = np.bincount(values)
  countFiltered = np.nonzero(countPerValue)[0]
  return list(zip(countFiltered, countPerValue[countFiltered]))

