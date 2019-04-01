
class Node:

    def __init__(self, catDict = None, classDict = None):
        self.cat = None
        self.threshold = None
        self.gain = None
        self.samples = None
        self.mostCommonValue = None
        self.percentage = None
        self.nodeClass = None
        self.false_branch = None
        self.true_branch = None
        self.countOfEach = None
        self.catDict = catDict
        self.classDict = classDict
        self.time = None

    def __str__(self):
        self.stringNode(0)
        return ""

    # =====Classifiers======
    def classify(self, catDict):
        if self.false_branch == None:
            return self.nodeClass
        catValue = catDict[self.cat]
        if catValue < self.threshold:
            return self.false_branch.classify(catDict)
        else:
            return self.true_branch.classify(catDict)

    def classifyPercent(self, catDict):
        if self.false_branch == None:
            return self.nodeClass, self.percentage
        catValue = catDict[self.cat]
        if catValue < self.threshold:
            return self.false_branch.classifyPercent(catDict)
        else:
            return self.true_branch.classifyPercent(catDict)

    # =====Printers=====
    def stringNode(self,n):
        ROUND = 3

        if not self.false_branch == None:
            if self.catDict == None:
                print(" "*n+"=>", "{} >= {}, Gain: {}".format(self.cat, round(self.threshold, ROUND), round(self.gain, ROUND) if not self.gain == None else "None"), ", MCV: ", self.mostCommonValue, ", P: ", round(self.percentage, ROUND), '% --- COE: ', self.countOfEach)
            else:
                print(" "*n+"=>", "{}({}) >= {}, Gain: {}".format(self.catDict[self.cat],self.cat, round(self.threshold, ROUND), round(self.gain, ROUND) if not self.gain == None else "None"), ", MCV: ", self.mostCommonValue, ", P: ", round(self.percentage, ROUND), '% --- COE: ', self.countOfEach)
            self.false_branch.stringNode(n+1)
            self.true_branch.stringNode(n+1)
        else:
            if self.classDict == None:
                print(" "*n+"->", "Class: {}".format(self.nodeClass), ", MCV: ", self.mostCommonValue, ", P: ", round(self.percentage, ROUND), '% --- COE: ', self.countOfEach)
            else:
                print(" "*n+"->", "Class: {}({})".format(self.classDict[self.nodeClass],self.nodeClass), ", MCV: ", self.mostCommonValue, ", P: ", round(self.percentage, ROUND), '% --- COE: ', self.countOfEach)

    def setTime(self, time):
        self.time = time