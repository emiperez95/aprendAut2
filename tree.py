import numpy as np
from functools import reduce

class Tree(object):
  def __init__(self):
      self.true_branch = None
      self.false_branch = None
      self.label = None

  def __str__(self):
    return self.data + ', ' + str(self.left) + ', ' + str(self.right)

class tree:
  def __init__ (self,trainingData):
    self.entropy = float(self.__shannonEntropy(trainingData))
    self.lengthTS = float(len(trainingData))

    self.train(trainingData)

  def __shannonEntropy(self, examples):
    entropy = 0
    length = float(len(examples)) # Si abajo en Pi = .. uno de los dos no es float, da 0 como resultado por ser ints
    counts = self.__countOfEachClass(examples)
    for count in counts:
      Pi = count[1]/length
      entropy -= Pi*np.log2(Pi)
    return entropy

  def train(self, data):
      # minInfoGain = 0
      # minInfoGainVal = -1
      # for val,col in enumerate(data.transpose()):
      #     infoGain = self.clasificarCol(col,clase)
      #     if minInfoGain < infoGain:
      #         minInfoGain = infoGain
      #         minInfoGainVal = val #TODO:

    # print(self.__gainAndThreshold(data, 0))
      # self.__id3(data, 4, [0,1,2,3])

  def __id3(self, examples, target_attribute, attributes):
    newRootNode = Tree()
    countOfEachClass = self.__countOfEachClass(examples)
    if (len(countOfEachClass) == 1):
      newRootNode.label = countOfEachClass[0][1]
      return newRootNode
    if (len(attributes) == 0):
      newRootNode.label = self.__mostCommonValueFor(examples, target_attribute) #TODO:
      return newRootNode
    bestAttribute, threshold = self.__bestFitAttribute(examples, attributes)


  def __bestFitAttribute(self, examples, attributes):
    # Si entra aca es porque hay attributes y al menos 2 ejemplos con distinta clase
    # Sino hubiese entrado en un caso base.
    bestIG = None
    bestThreshold = None
    bestAttribute = None
    for attr in attributes:
      IG, threshold = self.__gainAndThreshold(examples, attr)
      if bestIG == None or bestIG < IG:
        bestIG = IG
        bestThreshold = threshold
        bestAttribute = attr
    return bestAttribute, threshold

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

    return bestIG, bestThreshold

  def __partition(self, examples, attribute, threshold):
    condArr = examples[:,attribute] < threshold
    return examples[condArr], examples[~condArr]

  def __countOfEachClass(self, examples):
    values = examples[:,-1].astype(np.int)
    countPerValue = np.bincount(values)
    countFiltered = np.nonzero(countPerValue)[0]
    return list(zip(countFiltered, countPerValue[countFiltered]))
