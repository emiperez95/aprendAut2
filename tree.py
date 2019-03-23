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
    self.entropy = self.__shannonEntropy(trainingData)
    self.lengthTS = len(trainingData)

    self.train(trainingData)

  def __shannonEntropy(self, examples):
    entropy = 0
    length = float(len(examples))
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

      print(self.gainAndThreshold(data, 0))
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
    bestAttribute, thresholds = self.__bestFitAttribute(examples, attributes)


  def __bestFitAttribute(self, examples, attributes):
    # Si entra aca es porque hay al menos 2 ejemplos con distinta clase
    # Sino hubiese entrado en un caso base.
    pass

  def gainAndThreshold(self, examples, attribute):
    # Lets get the possible thresholds
    sortedExamples = np.sort(examples, axis=attribute)
    possibleThresholds = []
    lastClass = float(sortedExamples[0][-1])
    lastAttrValue = float(sortedExamples[0][attribute])
    for row in sortedExamples[1:]:
      if row[-1] != lastClass:
        possibleThresholds.append((float(row[attribute]) - lastAttrValue)/2)
      lastClass = float(row[-1])
      lastAttrValue = float(row[attribute])

    # Lets get the best threshold
    bestThreshold = None
    bestIG = None
    for threshold in possibleThresholds:
      partitionLess, partitionEqualGreat = self.__partition(examples, attribute, threshold)

      # H(T)
      TEntropy = self.entropy

      # IG(T, a) = H(T) - H(T|a)
      # H(T|a) = para todo v posible de vals(a) SUM((|Sa(v)|/|T|)*H(Sa(v)))
      IG = TEntropy
      - (len(partitionLess)/self.lengthTS)*self.__shannonEntropy(partitionLess)
      - (len(partitionEqualGreat)/self.lengthTS)*self.__shannonEntropy(partitionEqualGreat)

      if (bestIG == None or bestIG < IG):
        bestThreshold = threshold
        bestIG = IG

    return bestIG, bestThreshold


  def __partition(self, examples, attribute, threshold):
    attrCol = examples[:,attribute].astype(np.float)
    condArr = attrCol < threshold
    return attrCol[condArr], attrCol[~condArr]

  def __countOfEachClass(self, examples):
    values = examples[:,-1].astype(np.int)
    countPerValue = np.bincount(values)
    countFiltered = np.nonzero(countPerValue)[0]
    return list(zip(countFiltered, countPerValue[countFiltered]))















  def clasificarCol(self,col, clase):
      # Establezco como clasificar
      minimo = np.min(col)
      maximo = np.max(col)
      umbral = (maximo - minimo)/2
      return self.infoGain(col, clase, umbral)

  def infoGain(self, col, clase, umbral):
      bajo, alto, dictBajo, dictAlto = self.cantidadUmbral(col,clase,umbral)
      total = len(col)

      propBajo = bajo/total
      propAlto = alto/total

      entropiaTotalBajo = 0
      entropiaTotalAlto = 0

      for val in dictBajo.values():
          prop = val/bajo
          entropiaTotalBajo -=  prop * np.log(prop)

      for val in dictAlto.values():
          prop = val/alto
          entropiaTotalAlto -=  prop * np.log(prop)

      entropiaTotal = propBajo * entropiaTotalBajo + propAlto * entropiaTotalAlto

      return entropiaTotal

  def cantidadUmbral(self, col,clase, umbral):
      bajo = 0
      alto = 0
      dictBajo = {}
      dictAlto = {}
      for ind,cell in enumerate(col):
          if cell > umbral:
              if clase[ind] in dictBajo:
                  dictBajo[clase[ind]] += 1
              else:
                  dictBajo[clase[ind]] = 1
              bajo += 1

          else:
              if clase[ind] in dictAlto:
                  dictAlto[clase[ind]] += 1
              else:
                  dictAlto[clase[ind]] = 1
              alto += 1
      return bajo, alto, dictBajo, dictAlto
