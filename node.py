
class Node:

    def __init__(self, cat ,threshold, gain, samples, mostLikelyClass, leftChild = None, rightChild = None):
        self.cat = cat
        self.threshold = threshold
        self.gain = gain
        self.samples = samples
        self.mostLikelyClass = mostLikelyClass
        self.leftChild = leftChild
        self.rightChild = rightChild
    
    def __str__(self):
        self.stringNode(0)
        return "" 

    def classify(self, catDict):
        if self.leftChild == None:
            return self.mostLikelyClass
        catValue = catDict[self.cat]
        if catValue < self.threshold:
            return self.leftChild.classify(catDict)
        else:
            return self.rightChild.classify(catDict)

    # =====Printer=====
    def stringNode(self,n):
        print("  "*n+"-", "{}<{}, Gain: {}".format(self.cat, self.threshold, self.gain))
        self.leftChild.stringNode(n+1)
        self.rightChild.stringNode(n+1)
        return ""
