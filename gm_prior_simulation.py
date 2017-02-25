'''
Cai Shaofeng - 2017.2
Implementation of the Simulator
'''

import numpy as np
import matplotlib.pyplot as plt
import pickle

class Simulator():
    def __init__(self, gm_num, dimension, sample_num, pi, variance, covariace):
        # init simulator parameters
        self.gm_num, self.dimension, self.sample_num = gm_num, dimension, sample_num
        self.pi, self.variance, self.covariance = pi, variance, covariace

    def wGenerator(self):
        # generate w_origin from mixture coefficient
        self.w_origin = np.random.choice(4, size=(self.dimension), p=self.pi)
        self.w = np.ndarray(shape=(self.dimension), dtype='float32')

        # generate w with corresponding gaussian variance
        gm_count = np.bincount(self.w_origin)
        for gm_index in xrange(self.gm_num):
            self.w[self.w_origin==gm_index] = np.random.normal(0.0, np.sqrt(self.variance[gm_index]), size=(gm_count[gm_index]))

    def xGenerator(self):
        self.x = np.random.multivariate_normal(mean=np.zeros(shape=(self.dimension)), cov=self.covariance, size=(self.sample_num,))

    def labelGenerator(self, noiseVar=0.1):
        lg, noise = np.dot(self.x, self.w), np.random.normal(0.0, np.sqrt(noiseVar), size=(self.sample_num))
        y_vals_no_noise =  1/(1+np.exp(-lg))
        y_vals = 1/(1+np.exp(-(lg+noise)))    # adding the gaussian noise term

        uniform_vals = np.random.uniform(low=0.0, high=1.0, size=(self.sample_num))
        self.label_no_noise = (y_vals_no_noise >= uniform_vals)
        self.label = (y_vals>=uniform_vals)

        # save generated y_vals_noise
        with open('generated_y_vals.dta', 'w') as saveFile:
            for index, item in enumerate(self.label):
                saveFile.write("%d: %.6f > %.6f\t(yvals %.6f,\tyvals_noise %.6f,\tnoise %10.6f)\n"
                               %(item, y_vals[index], uniform_vals[index], y_vals_no_noise[index], y_vals[index], noise[index]))

        # get probability distribution and noise effect
        print "probability distribution:\t", (np.bincount((y_vals_no_noise*10).astype(int)).astype(float)/self.sample_num)
        print "noise misclassification rate:\t%.6f" %(float(np.sum(self.label!=self.label_no_noise))/self.sample_num)

def generateCov(n=5, a=1, showCov=True):
    # see http://stats.stackexchange.com/questions/124538/how-to-generate-a-large-full-rank-random-correlation-matrix-with-some-strong-cor
    A = np.matrix([np.random.randn(n) + np.random.randn(1) * a for i in range(n)])
    A = A * np.transpose(A)
    D_half = np.diag(np.diag(A) ** (-0.5))
    C = D_half * A * D_half
    # show the cov matrix
    if showCov:
        vals = list(np.array(C.ravel())[0])
        plt.hist(vals, range=(-1,1))
        plt.show()
        plt.imshow(C, interpolation=None)
        plt.show()
    return C


if __name__ == '__main__':
    gm_num, dimension, sample_num = 4, 1000, 10000
    pi, variance, covariance = np.array([0.4, 0.3, 0.2, 0.1]), np.array([0.001, 0.01, 0.02, 0.1]), generateCov(n=dimension, a=1, showCov=False)

    simulator = Simulator(gm_num, dimension, sample_num, pi, variance, covariance)
    simulator.wGenerator()
    simulator.xGenerator()
    simulator.labelGenerator(noiseVar=0.5)

    # save the generated simulator
    with open('simulator.pkl', 'w') as saveFile:
        pickle.dump(simulator, saveFile)

'''
generated results see generated_y_vals.dta
simulator file saved in simulator.pkl

using pkl data:
import pickle
from gm_prior_simulation import Simulator
pickle.load(open('simulator.pkl', 'r'))

probability distribution:	[ 0.2383  0.0896  0.0594  0.0546  0.049   0.0509  0.0536  0.0651  0.0872
  0.2523]
noise misclassification rate:	0.057400
'''




