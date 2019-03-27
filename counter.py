import numpy as np

class Counter:
    def __init__(self):
        self.count = 0
        self.timeArr = []
        self.lnArr = []
    
    def __str__(self):
        arr = np.array(self.timeArr)
        arr2 = np.array(self.lnArr)
        return "Cantidad de entradas: {}, tiempo promedio: {}, largo medio: {}, tiempoTotal: {}, largoTotal: {}".format(self.count, np.mean(arr), np.mean(arr2), np.sum(arr), np.sum(arr2))

    def oper(self, time, ln):
        self.count += 1
        self.timeArr.append(time)
        self.lnArr.append(ln)
