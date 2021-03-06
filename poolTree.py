from tree import makeNode
from node import Node
import numpy as np


class PoolTree:
    def __init__(self, allData, classCount, catAmm, partitionStyle = True, entropyFunc = 0, attTypes = None, repetition = 0, entThresh=0.0):
        self.time = None
        dataArr = [[] for _ in range(classCount)]
        for row in allData:
            rowClass = row[-1] -1
            rowFalse = row.copy()
            rowFalse[-1] = 0
            rowTrue = row.copy()
            rowTrue[-1] = 1

            for i in range(classCount):
                if int(rowClass) == i:
                    dataArr[i].append(rowTrue)
                else:
                    dataArr[i].append(rowFalse)
        for i in range(len(dataArr)):
            dataArr[i] = np.array(dataArr[i])

        self.nodeArr = [makeNode(data, catAmm, partitionStyle=False, entropyFunc=0, catTypeArr=attTypes, repetition = repetition, entThresh = entThresh) for data in dataArr]
        
    def __str__(self):
        for node in self.nodeArr:
            print(node)
        return ""

    def classify(self, row):
        resArr = []
        elseArr = []
        for i, node in enumerate(self.nodeArr):
            rowClass, percent = node.classifyPercent(row)
            if rowClass == 1:
                resArr.append((i, percent))
            else:
                elseArr.append((i, percent))
        
        result = 1
        #Clasificador con porcentajes
        if resArr == []:
            minRes = elseArr[0][0]
            minResVal = elseArr[0][1]
            for a in elseArr[1:]:
                if a[1] < minResVal:
                    minResVal = a[1]
                    minRes = a[0]
            result = minRes
        elif len(resArr) > 1:
            maxRes = resArr[0][0]
            maxResVal = resArr[0][1]
            for a in resArr[1:]:
                if a[1] > maxResVal:
                    maxResVal = a[1]
                    maxRes = a[0]
            result = maxRes
        else:
            result = resArr[0][0]
        return result + 1
        
    def setTime(self, time):
        self.time = time
