import numpy as np
# from matplotlib import pyplot as plt
import pickle
from gm_prior_simulation import Simulator


# load training/testing data from pickles simulator object file with specified training percent
def loadData(simulatorPath, trainPerc=0.7):
    print '\n===============================================\n'
    print 'loading data...'
    simulator = pickle.load(open(simulatorPath, 'r'))
    trainNum = int(simulator.sample_num * trainPerc)
    simulator.label.resize((simulator.sample_num, 1))
    # prepare training/testing data
    xTrain, xTest = simulator.x[:trainNum], simulator.x[trainNum:]
    yTrain, yTest = simulator.label[:trainNum], simulator.label[trainNum:]
    print 'finish loading data...\ntraining data samples %d\ntesting data samples %d' %(trainNum, simulator.sample_num-trainNum)
    print 'data dimension %d' %(simulator.dimension)
    print '\n===============================================\n'

    return xTrain, xTest, yTrain, yTest

