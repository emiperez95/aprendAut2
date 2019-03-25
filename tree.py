import numpy as np
from functools import reduce
from node import Node


def makeNode (trainingData, catDict = None, classDict = None):
  varDict = {
    "entropy" : float(__entropy(trainingData)),
    "lengthTS" : float(len(trainingData)),
    "catDict" : catDict,
    "classDict" : classDict
  }
  return __id3(varDict, trainingData, 4, [0,1,2,3])

def __entropy(examples):
  return __giniImpurity(examples)

def __shannonEntropy(examples):
  entropy = 0
  length = float(len(examples)) # Si abajo en Pi = .. uno de los dos no es float, da 0 como resultado por ser ints
  counts = __countOfEachClass(examples)
  for count in counts:
    Pi = count[1]/length
    entropy -= Pi*np.log2(Pi)
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

def __id3(varDict, examples, target_attribute, attributes):
  newRootNode = Node(varDict["catDict"], varDict["classDict"])
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
  bestAttribute, threshold, partitionLess, partitionEqualGreat, bestIG  = __bestFitAttribute(varDict, examples, attributes)
  newAttributesSet = list(set(attributes) - {bestAttribute})
  newRootNode.cat = bestAttribute
  newRootNode.threshold = threshold
  newRootNode.gain = bestIG
  if (len(partitionLess) == 0):
    newRootNode.false_branch = Node(varDict["catDict"], varDict["classDict"])
    newRootNode.false_branch.nodeClass = mostCommonValue
    newRootNode.false_branch.mostCommonValue = mostCommonValue
    newRootNode.false_branch.percentage = percentage
    newRootNode.false_branch.countOfEach = []
  else:
    newRootNode.false_branch = __id3(varDict, partitionLess, target_attribute, newAttributesSet)
  if (len(partitionEqualGreat) == 0):
    newRootNode.true_branch = Node(varDict["catDict"], varDict["classDict"])
    newRootNode.true_branch.nodeClass = mostCommonValue
    newRootNode.true_branch.mostCommonValue = mostCommonValue
    newRootNode.true_branch.percentage = percentage
    newRootNode.true_branch.countOfEach = []
  else:
    newRootNode.true_branch = __id3(varDict, partitionEqualGreat, target_attribute, newAttributesSet)
  return newRootNode

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
    IG, threshold, partitionLess, partitionEqualGreat = __gainAndThreshold(varDict, examples, attr)
    if bestIG == None or bestIG < IG:
      bestIG = IG
      bestThreshold = threshold
      bestAttribute = attr
      bestPartitionLess = partitionLess
      bestPartitionEqualGreat = partitionEqualGreat
  return bestAttribute, bestThreshold, partitionLess, partitionEqualGreat, bestIG

def __gainAndThreshold(varDict, examples, attribute):
  # Lets get the possible thresholds
  sortedExamples = examples[np.argsort(examples[:,attribute])]
  possibleThresholds = []
  lastClass = sortedExamples[0][-1]
  lastAttrValue = sortedExamples[0][attribute]
  for row in sortedExamples[1:]:
    if row[-1] != lastClass:
      possibleThresholds.append((row[attribute] + lastAttrValue)/2)
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
    partitionLess, partitionEqualGreat = __partition(examples, attribute, threshold)
    # H(T)
    TEntropy = varDict["entropy"]

    # IG(T, a) = H(T) - H(T|a)
    # H(T|a) = para todo v posible de vals(a) SUM((|Sa(v)|/|T|)*H(Sa(v)))
    HLess = (len(partitionLess)/varDict["lengthTS"])*__entropy(partitionLess)
    HEqualGreat = (len(partitionEqualGreat)/varDict["lengthTS"])*__entropy(partitionEqualGreat)
    IG = TEntropy - HLess - HEqualGreat

    if (bestIG == None or bestIG < IG):
      bestThreshold = threshold
      bestIG = IG
      bestPartitionLess = partitionLess
      bestPartitionEqualGreat = partitionEqualGreat

  return bestIG, bestThreshold, bestPartitionLess, bestPartitionEqualGreat

def __partition(examples, attribute, threshold):
  condArr = examples[:,attribute] <= threshold
  return examples[condArr], examples[~condArr]

def __countOfEachClass(examples):
  values = examples[:,-1].astype(np.int)
  countPerValue = np.bincount(values)
  countFiltered = np.nonzero(countPerValue)[0]
  return list(zip(countFiltered, countPerValue[countFiltered]))

