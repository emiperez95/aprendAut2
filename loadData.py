import numpy as np
from tree import makeNode
from poolTree import PoolTree
import random
import time
import pickle

import binarytree as bt

def nodeToBtNode(nodo):
    if nodo.false_branch == None:
        btNode = bt.Node(nodo.mostCommonValue)
    else:
        btNode = bt.Node(nodo.cat)
        btNode.right = nodeToBtNode(nodo.false_branch)
        btNode.left = nodeToBtNode(nodo.true_branch)

    return btNode

dataDump = "persist/temp2.th"
with open(dataDump, 'rb') as fl:
    modelo = pickle.load(fl)

# print(modelo)
btNode = nodeToBtNode(modelo)
print(btNode)