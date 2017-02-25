'''
Cai Shaofeng - 2017.2
Implementation of the Logistic Regression
'''
# SGDClassifier for Logistic Regression
from sklearn import linear_model
from data_loader import *
from sklearn.metrics import accuracy_score, roc_auc_score
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
        #grad_w = np.matmul(xTrain.T, (mu - yTrain)) + self.reg_lambda * self.w
        S = np.diag((mu*(1-mu)).reshape((xTrain.shape[0])))
        hessian = np.matmul(xTrain.T, np.matmul(S, xTrain)) + self.reg_lambda*np.identity(self.w.shape[0])
        return -np.matmul(np.linalg.pinv(hessian), grad_w)

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
    xTrain, xTest, yTrain, yTest = loadData('simulator_1_gaussian.pkl', trainPerc=0.7)

    train_accuracy, test_accuracy =  [], []

    # create logistic regression class
    # reg_lambda, learning_rate, max_iter, eps, batch_size = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 0.1, 50, 1e-3, 500
    reg_lambda, learning_rate = [0.001, 1, 10, 100], 0.1
    for reg in reg_lambda:
        clf = linear_model.SGDClassifier(alpha=reg, learning_rate='constant', n_iter=5, eta0=learning_rate, loss='log')
        clf = linear_model.SGDClassifier(alpha=reg, learning_rate='optimal', n_iter=5, eta0=learning_rate, loss='log')
        clf.fit(xTrain, yTrain)
        print "reg: ", reg
        print "accuracy: ", accuracy_score(yTest, clf.predict(xTest))
