
class Node:

    def __init__(self, catDict = None, classDict = None):
        self.cat = None
        self.threshold = None
        self.gain = None
        self.samples = None
        self.nodeClass = None
        self.false_branch = None
        self.true_branch = None
        self.catDict = catDict
        self.classDict = classDict
    
    def __str__(self):
        self.stringNode(0)
        return "" 

    def classify(self, catDict):
        if self.false_branch == None:
            return self.nodeClass
        catValue = catDict[self.cat]
        if catValue < self.threshold:
            return self.false_branch.classify(catDict)
        else:
            return self.true_branch.classify(catDict)
    
    # =====Printer=====
    def stringNode(self,n):
        if not self.false_branch == None:
            if self.catDict == None:
                print("  "*n+"=>", "{} >= {}, Gain: {}".format(self.cat, self.threshold, round(self.gain, 3) if not self.gain == None else "None"))
            else:
                print("  "*n+"=>", "{}({}) >= {}, Gain: {}".format(self.catDict[self.cat],self.cat, self.threshold, round(self.gain, 3) if not self.gain == None else "None"))
            self.false_branch.stringNode(n+1)
            self.true_branch.stringNode(n+1)
        else:
            if self.classDict == None:
                print("  "*n+"=>", "Class: {}".format(self.nodeClass))
            else:
                print("  "*n+"=>", "Class: {}({})".format(self.classDict[self.nodeClass],self.nodeClass))
