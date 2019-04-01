import numpy as np
import pickle
import sys
from tree import makeNode
from poolTree import PoolTree
from evaluation import Evaluation
import sys
sys.setrecursionlimit(200000)

CLASS_AMM = 7
TEST_DATA_PATH = "./data/evaDataCover.npy"
DIR_MODELO = sys.argv[1]

testData = np.load(TEST_DATA_PATH)

with open(DIR_MODELO, 'rb') as fl:
    model = pickle.load(fl)


eval = Evaluation(model, testData, CLASS_AMM)
eval.normalPrint()

