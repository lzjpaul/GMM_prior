'''
Cai Shaofeng - 2017.2
Implementation of the Logistic Regression
'''

from data_loader import *

# base logistic regression class
class Logistic_Regression():
    def __init__(self, reg_lambda=1, learning_rate=0.1, max_iter=1000, eps=1e-4, batch_size=-1, validation_perc=0.3):
        self.reg_lambda, self.learning_rate, self.max_iter = reg_lambda, learning_rate, max_iter
        self.eps, self.batch_size, self.validation_perc = eps, batch_size, validation_perc

    # calc the delta w to update w, using newton's method here
    def delta_w(self, xTrain, yTrain):
        # mini batch, not used here
        if self.batch_size != -1:
            randomIndex = np.random.random_integers(0, xTrain.shape[0]-1, self.batch_size)
            xTrain, yTrain = xTrain[randomIndex], yTrain[randomIndex]

        mu = self.sigmoid(np.matmul(xTrain, self.w))
        # check here, no regularization over bias term
        grad_w = np.matmul(xTrain.T, (mu-yTrain)) + np.vstack(([0], np.full((self.w.shape[0]-1,1), self.reg_lambda, dtype='float32')))*self.w
        grad_w = np.matmul(xTrain.T, (mu - yTrain)) + self.reg_lambda * self.w
        S = np.diag((mu*(1-mu)).reshape((xTrain.shape[0])))
        hessian = np.matmul(xTrain.T, np.matmul(S, xTrain)) + self.reg_lambda*np.identity(self.w.shape[0])
        return -np.matmul(np.linalg.pinv(hessian), grad_w)
        #return -grad_w

    def fit(self, xTrain, yTrain):
        # find the number of class and feature, allocate memory for model parameters
        self.trainNum, self.featureNum = xTrain.shape[0], xTrain.shape[1]
        self.w = np.random.normal(0, 0.0001, size=(self.featureNum+1, 1)) #np.zeros(shape=(self.featureNum+1, 1), dtype='float32')#np.random.normal(0, 1, size=(self.featureNum+1, 1))

        # adding 1s to each training examples
        xTrain = np.hstack((np.ones(shape=(self.trainNum, 1)), xTrain))

        # validation set
        validationNum = int(self.validation_perc*xTrain.shape[0])
        xVallidation, yVallidation = xTrain[:validationNum, ], yTrain[:validationNum, ]
        xTrain, yTrain = xTrain[validationNum:, ], yTrain[validationNum:, ]

        try:
            iter, self.best_accuracy, self.best_iter = 0, 0.0, 0
            while True:
                # calc the delta_w to update w
                delta_w = self.delta_w(xTrain, yTrain)
                # update w
                self.w += self.learning_rate * delta_w

                # stop updating if converge
                iter += 1
                if iter > self.max_iter or np.linalg.norm(delta_w, ord=2) < self.eps:
                    break

                test_accuracy = self.accuracy(self.predict(xVallidation), yVallidation)
                train_accuracy = self.accuracy(self.predict(xTrain), yTrain)
                if self.best_accuracy < test_accuracy:
                    self.best_accuracy, self.best_iter = test_accuracy, iter
                # print "iter %4d\t|\ttrain_accuracy %10.6f\t|\ttest_accuracy %10.6f\t|\tbest_accuracy %10.6f"\
                #       %(iter, train_accuracy, test_accuracy, self.best_accuracy)
        except:
            pass

    # predict result
    def predict(self, samples):
        if samples.shape[1] != self.w.shape[0]:
            samples = np.hstack((np.ones(shape=(samples.shape[0], 1)), samples))
        return np.matmul(samples, self.w)>0.5

    # calc accuracy
    def accuracy(self, yPredict, yTrue):
        return np.sum(yPredict == yTrue) / float(yTrue.shape[0])

    # sigmoid function
    def sigmoid(self, matrix):
        return 1.0/(1.0+np.exp(-matrix))

    # model parameter
    def __str__(self):
        return 'model parameter {\treg: %.6f, lr: %.6f, batch_size: %5d, best_iter: %6d, best_accuracy: %.6f\t}' \
            % (self.reg_lambda, self.learning_rate, self.batch_size, self.best_iter, self.best_accuracy)

if __name__ == '__main__':
    # load the simulation data
    xTrain, xTest, yTrain, yTest = loadData('simulator.pkl', trainPerc=0.7)

    train_accuracy, test_accuracy =  [], []

    # create logistic regression class
    reg_lambda, learning_rate, max_iter, eps, batch_size = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], 0.1, 50, 1e-3, 500
    for reg in reg_lambda:
        print "\nreg_lambda: %f" %(reg)
        LG = Logistic_Regression(reg, learning_rate, max_iter, eps, batch_size)
        LG.fit(xTrain, yTrain)
        train_accuracy.append(LG.best_accuracy)
        test_accuracy.append(LG.accuracy(LG.predict(xTest), yTest))
        print "finally accuracy: %.6f" %(test_accuracy[-1])
        print LG

    fig, ax = plt.subplots()
    ax.plot(reg_lambda, train_accuracy, 'r', label='train_accuracy'); ax.plot(reg_lambda, test_accuracy, 'b', label='test_accuracy');
    ax.set_xscale('log'); ax.set_xticks(reg_lambda); plt.xlabel('reg_lambda');plt.ylabel('accuracy');
    plt.legend(loc="lower left", ncol=1, shadow=True, fancybox=True)
    plt.title('accuracy VS reg_lambda'); plt.savefig('data/l2_accuracy.eps', format='eps', dpi=1000);
    plt.show()


'''
best reg_lamdba in range [0.1, 10], around 1, most likely to be [1, 10]

reg_lambda: 0.00001
finally accuracy: 0.603333
model parameter {	reg: 0.000010, lr: 0.100000, batch_size:   500, best_iter:     33, best_accuracy: 0.790000	}

reg_lambda: 0.00010
/home/shawn/Desktop/GM_prior/logistic_regression.py:76: RuntimeWarning: overflow encountered in exp
  return 1.0/(1.0+np.exp(-matrix))
finally accuracy: 0.706667
model parameter {	reg: 0.000100, lr: 0.100000, batch_size:   500, best_iter:     44, best_accuracy: 0.793810	}

reg_lambda: 0.00100
finally accuracy: 0.805333
model parameter {	reg: 0.001000, lr: 0.100000, batch_size:   500, best_iter:     38, best_accuracy: 0.792381	}

reg_lambda: 0.01000
finally accuracy: 0.796333
model parameter {	reg: 0.010000, lr: 0.100000, batch_size:   500, best_iter:     27, best_accuracy: 0.781429	}

reg_lambda: 0.10000
finally accuracy: 0.800333
model parameter {	reg: 0.100000, lr: 0.100000, batch_size:   500, best_iter:     49, best_accuracy: 0.783333	}

reg_lambda: 1.00000
finally accuracy: 0.817000
model parameter {	reg: 1.000000, lr: 0.100000, batch_size:   500, best_iter:     35, best_accuracy: 0.794762	}

reg_lambda: 10.00000
finally accuracy: 0.805667
model parameter {	reg: 10.000000, lr: 0.100000, batch_size:   500, best_iter:     36, best_accuracy: 0.783333	}

reg_lambda: 100.00000
finally accuracy: 0.760000
model parameter {	reg: 100.000000, lr: 0.100000, batch_size:   500, best_iter:     30, best_accuracy: 0.748095	}

reg_lambda: 1000.00000
finally accuracy: 0.691667
model parameter {	reg: 1000.000000, lr: 0.100000, batch_size:   500, best_iter:     44, best_accuracy: 0.677143	}

reg_lambda: 10000.00000
finally accuracy: 0.645333
model parameter {	reg: 10000.000000, lr: 0.100000, batch_size:   500, best_iter:     42, best_accuracy: 0.631905	}

'''

