from tree import makeNode
from node import Node
import numpy as np


class PoolTree:
    def __init__(self, allData, classCount, clasificador = 1): #Suponemos data normalizada
        self.clasificador = clasificador

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


        self.nodeArr = [makeNode(data) for data in dataArr]
        
    def __str__(self):
        for node in self.nodeArr:
            print(node)
        return ""

    def classify(self, row):
        resArr = []
        for i, node in enumerate(self.nodeArr):
            rowClass = node.classify(row)
            if rowClass == 1:
                resArr.append(i)
        
        result = 1
        #Clasificador usado para mostrar mejorias en casos de disputas
        if self.clasificador == 0: 
            if resArr == []: #TODO: debe ser el de menor porcentaje
                pass
            elif len(resArr) > 1: #TODO: debe ser el de mejor porcentaje
                result = resArr[0] + 1
            else:
                result = resArr[0] + 1
        else: #TODO: implementar otro clasificador
            if resArr == []: #TODO: debe ser el de menor porcentaje
                pass
            elif len(resArr) > 1: #TODO: debe ser el de mejor porcentaje
                result = resArr[0] + 1
            else:
                result = resArr[0] + 1
        return result
        

