import numpy as np
from functools import reduce
from node import Node

class tree:
  def __init__ (self,trainingData, catDict = None, classDict = None):
    self.entropy = float(self.__shannonEntropy(trainingData))
    self.lengthTS = float(len(trainingData))
    self.catDict = catDict
    self.classDict = classDict
    self.node = self.__id3(trainingData, 4, [0,1,2,3])

  def getNode(self):
    return self.node

  def __shannonEntropy(self, examples):
    entropy = 0
    length = float(len(examples)) # Si abajo en Pi = .. uno de los dos no es float, da 0 como resultado por ser ints
    counts = self.__countOfEachClass(examples)
    for count in counts:
      Pi = count[1]/length
      entropy -= Pi*np.log2(Pi)
    return entropy

  def __id3(self, examples, target_attribute, attributes):
    newRootNode = Node(self.catDict, self.classDict)
    countOfEachClass = self.__countOfEachClass(examples)
    mostCommonValue = max(countOfEachClass, key=lambda item: item[1])[0]
    if (len(countOfEachClass) == 1):
      newRootNode.nodeClass = countOfEachClass[0][0]
      return newRootNode
    if (len(attributes) == 0):
      newRootNode.nodeClass = mostCommonValue
      return newRootNode
    bestAttribute, threshold, partitionLess, partitionEqualGreat, bestIG  = self.__bestFitAttribute(examples, attributes)
    newAttributesSet = list(set(attributes) - {bestAttribute})
    newRootNode.cat = bestAttribute
    newRootNode.threshold = threshold
    newRootNode.gain = bestIG
    if (len(partitionLess) == 0):
      newRootNode.false_branch = Node(self.catDict, self.classDict)
      newRootNode.false_branch.nodeClass = mostCommonValue
    else:
      newRootNode.false_branch = self.__id3(partitionLess, target_attribute, newAttributesSet)
    if (len(partitionEqualGreat) == 0):
      newRootNode.true_branch = Node(self.catDict, self.classDict)
      newRootNode.true_branch.nodeClass = mostCommonValue
    else:
      newRootNode.true_branch = self.__id3(partitionEqualGreat, target_attribute, newAttributesSet)
    return newRootNode


  def __bestFitAttribute(self, examples, attributes):
    # Si entra aca es porque hay attributes y al menos 2 ejemplos con distinta clase
    # Sino hubiese entrado en un caso base.
    bestIG = None
    bestThreshold = None
    bestAttribute = None
    bestPartitionLess = None
    bestPartitionEqualGreat = None
    for attr in attributes:
      IG, threshold, partitionLess, partitionEqualGreat = self.__gainAndThreshold(examples, attr)
      if bestIG == None or bestIG < IG:
        bestIG = IG
        bestThreshold = threshold
        bestAttribute = attr
        bestPartitionLess = partitionLess
        bestPartitionEqualGreat = partitionEqualGreat
    return bestAttribute, bestThreshold, partitionLess, partitionEqualGreat, bestIG

  def __gainAndThreshold(self, examples, attribute):
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
      partitionLess, partitionEqualGreat = self.__partition(examples, attribute, threshold)
      # H(T)
      TEntropy = self.entropy

      # IG(T, a) = H(T) - H(T|a)
      # H(T|a) = para todo v posible de vals(a) SUM((|Sa(v)|/|T|)*H(Sa(v)))
      HLess = (len(partitionLess)/self.lengthTS)*self.__shannonEntropy(partitionLess)
      HEqualGreat = (len(partitionEqualGreat)/self.lengthTS)*self.__shannonEntropy(partitionEqualGreat)
      IG = TEntropy - HLess - HEqualGreat

      if (bestIG == None or bestIG < IG):
        bestThreshold = threshold
        bestIG = IG
        bestPartitionLess = partitionLess
        bestPartitionEqualGreat = partitionEqualGreat

    return bestIG, bestThreshold, bestPartitionLess, bestPartitionEqualGreat

  def __partition(self, examples, attribute, threshold):
    condArr = examples[:,attribute] < threshold
    return examples[condArr], examples[~condArr]

  def __countOfEachClass(self, examples):
    values = examples[:,-1].astype(np.int)
    countPerValue = np.bincount(values)
    countFiltered = np.nonzero(countPerValue)[0]
    return list(zip(countFiltered, countPerValue[countFiltered]))

