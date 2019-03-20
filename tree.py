import numpy as np

class tree:
    def __init__ (self,trainingData):
        self.train(trainingData[:,:-1], trainingData[:,-1])

    def train(self,data,clase):
        minInfoGain = 0
        minInfoGainVal = -1
        for val,col in enumerate(data.transpose()):
            infoGain = self.clasificarCol(col,clase)
            if minInfoGain < infoGain:
                minInfoGain = infoGain
                minInfoGainVal = val #TODO:

        pass

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
