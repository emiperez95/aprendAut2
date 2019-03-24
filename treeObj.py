
class node:
    def __init__(self,isLeaf, name, contents = None):
        self.isLeaf = isLeaf
        if isLeaf:
            self.name = name
        else:
            self.name = name
            self.catDict = {}
            for cat in contents:
                self.catDict[cat] = None
    
    def __str__(self):
        self.stringNode(0)
        return ""

    # =====Getters=====
    def getName(self):
        return self.name

    def getCatDict(self):
        if self.isLeaf:
            raise Exception("Can't call getCatDict on a Leaf node: {}".format(self.name))
        else:
            return self.catDict

    def getIsLeaf(self):
        return self.isLeaf    
    
    #=====Setters=====
    def setSonCat(self, cat, son):
        if self.isLeaf:
            raise Exception("Can't set Son to a Leaf node: {}".format(self.name))
        else:
            if cat in self.catDict:
                self.catDict[cat] = son
            else:
                raise Exception("{} has no category: {}".format(self.name, cat))
    
    # =====Printer=====
    def stringNode(self,n):
        print(" "*n+"-", self.name)
        for cat in self.catDict:
            print(" "*n+1+".", cat)
            son = self.catDict[cat]
            son.strinNode(n+2)
        return ""
