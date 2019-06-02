import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class SolveMinProb:
    def __init__(self, y = np.ones((3,1)), A = np.eye(3)):
        self.matr = A
        self.Np = y.shape[0]
        self.Nf = A.shape[1]
        self.vect = y
        self.sol = np.zeros((self.Nf, 1), dtype = float)
        return
    def plot_w (self, title = 'Solution'):
        w = self.sol
        n = np.arange(self.Nf)
        plt.figure()
        plt.stem(n,w)
        plt.xlabel('features')
        plt.ylabel('w for each feature')
        plt.ylim(ymin = -0.3, ymax = 1)
        plt.title(title)
        plt.grid()
        plt.show()
        return
    def plot_y(self, A_test, A_train, y_test, y_train,  mean, st_dev, title):
        '''
        :param A_test: matrix of test data set
        :param w: is the optimum weight vector
        :param A_train: matrix of train data set. This is used to find w
        :param y_test: column taken from the matrix A_test, denormalized
        :param y_train: column taken from the matrix A_train, denormalized
        :param y_hat_test:calculated with matrix at which each vector belongs times w, the weight vector. We actually know what the true y_test is and so we can measure the estimation error on the testing data e_test = y_test - y_hat_test
        and then we can calculate the mean square error for the testing data MSE_test = ||e_test||^2/N_test(rows)
        :param mean: row vector: each element is the mean calculated for each column of the  matrix containing all data
        :param st_dev: vector of mean calculated for each column of the  matrix containing all data
        :return: returns the plot of y and y_hat
        '''

        w = self.sol
        '''Here i denormalize the vector y'''
        y_hat_train = np.dot(A_train, w)*st_dev + mean
        y_hat_test = np.dot(A_test, w)*st_dev + mean
        y_train = y_train*st_dev + mean
        y_test = y_test*st_dev + mean
        #plt.plot(np.linspace(0, 50), np.linspace(0, 50), color ='black')

        slope,intercept,r_value,p_value,std_err= stats.mstats.linregress(y_hat_train,y_train)
        line = slope*y_hat_train+intercept
        plt.title(title)
        plt.figure()
        plt.plot(y_hat_train,line, color = 'black')
        plt.scatter(y_hat_train, y_train, s = 3)
        plt.xlabel('y_hat_train')
        plt.ylabel('y_train')
        plt.title(title)
        plt.grid()
        plt.show()


        plt.title(title)
        slope, intercept, r_value, p_value, std_err = stats.mstats.linregress(y_hat_test, y_test)
        line = slope * y_hat_test + intercept
        plt.title(title)
        plt.figure()
        plt.plot(y_hat_test, line, color = 'black')
        plt.scatter(y_hat_test, y_test,s = 3, color = 'orange')
        plt.xlabel('y_hat_test')
        plt.ylabel('y_test')
        plt.title(title)
        plt.grid()
        plt.show()

    def plot_hist(self, A_train, A_test, y_train, y_test, title):
        print(title, '')
        '''
        This method is used to plot the histograms of y_hat_train-y_hat and y_hat_test-y_test
        '''
        w = self.sol
        y_hat_train = np.dot(A_train, w)
        y_hat_test = np.dot(A_test, w)
        error_test = y_test - y_hat_test
        error_train = y_train - y_hat_train

        plt.hist(error_test, bins=50, color = 'orange')
        plt.xlabel('ŷ_test-y_test')
        plt.ylabel('number occurencies')
        plt.title(title)
        plt.grid()
        plt.xlim(xmin = -2, xmax = 2)
        plt.ylim(ymin = 0, ymax = 600)
        str = title+'.pdf'
        plt.savefig(str, bbox_inches = 'tight')
        plt.show()
        plt.hist(error_train, bins=50)  # arguments are passed to np.histogram
        plt.xlabel('ŷ_train-y_train')
        plt.ylabel('number occurences')
        plt.title(title)
        plt.grid()
        plt.xlim(xmin=-2, xmax=2)
        plt.ylim(ymin=0, ymax=600)
        plt.show()
        return
    def print_result(self,title):
        print(title, ' ')
        print('the optimum weight vector is: ')
        print(self.sol)
        return

    def plot_err(self,title = 'Mean Square error', logy = 0, logx = 0): #plotta l'errore square error
        ''' this method plots the Mean Square Error'''
        err = self.err
        plt.figure()
        if (logy == 0) & (logx == 0):
            plt.plot(err[:,0], err[:,1], label = 'train')
            plt.plot(err[:, 0], err[:, 2], label='val')
            #plt.plot(err[:, 0], err[:, 3], label='test')

        if (logy == 1) & (logx == 0):
            plt.semilogy(err[:,0],err[:,1], label = 'train')
            plt.semilogy(err[:, 0], err[:, 2], label = 'val')
            #plt.plot(err[:, 0], err[:, 3], label='test')

        if (logy == 0) & (logx == 1):
            plt.semilogx(err[:,0], err[:,1], label = 'train')
            plt.semilogx(err[:, 0], err[:, 2], label = 'val')
            #plt.plot(err[:, 0], err[:, 3], label='test')

        if (logy == 1) & (logx == 1):
            plt.loglog(err[:,0], err[:,1], label = 'train')
            plt.loglog(err[:, 0], err[:, 2], label = 'val')
            #plt.plot(err[:, 0], err[:, 3], label='test')

        plt.legend()
        plt.xlabel('n° of iteration')
        plt.ylabel('Mean Square Error')
        plt.title(title)
        plt.margins(0.01, 0.1)
        plt.legend()
        plt.grid()
        plt.xlim(xmin = 0, xmax = 100)
        plt.show()

class SolveLLS (SolveMinProb):
    def run(self, A_train, y_train, A_test, y_test, A_val, y_val):
        w = np.dot(np.linalg.pinv(A_train), y_train)
        #w = np.dot(np.dot(np.linalg.inv(np.dot(A_train.T, A_train)),A_train.T),y_train)
        self.sol = w
        self.min = np.linalg.norm(np.dot(A_train, w) - y_train)
        self.MSE_train = np.linalg.norm(np.dot(A_train, w) - y_train)**2/A_train.shape[0]
        self.MSE_test = np.linalg.norm(np.dot(A_test, w) - y_test)**2/A_test.shape[0]
        self.MSE_val = np.linalg.norm(np.dot(A_val, w) - y_val)**2/A_val.shape[0]
        print("MSE of Train")
        print(self.MSE_train)
        print("MSE of test")
        print(self.MSE_test)
        print("MSE of val")
        print(self.MSE_val)
        print("self min : ", self.min)


'''
For the iterative algorithms in order to evaluate the MSE it has been calculated in each 
iteration error_val (as y_val - y_hat_val), error_train (as y_train - y_hat_train) 
and error_test (as y_test - y_hat_test) and a matrix self.err has been uploaded with this values.
'''
class SolveRidge(SolveMinProb):
    """" Ridge Algorithm """
    def run(self, A_train, A_val, A_test, y_train, y_val, y_test):
        np.random.seed(3)
        # w = np.zeros
        w = np.random.rand(self.Nf, 1)
        I = np.eye(self.Nf)
        Nit = 300
        self.err = np.zeros((Nit, 4), dtype=float)
        for it in range(Nit):
            w = np.dot(np.dot(np.linalg.inv(np.dot(A_train.T, A_train)+float(it)*I), A_train.T), y_train)
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm(np.dot(A_train, w) - y_train)**2 / A_train.shape[0]
            self.err[it, 2] = np.linalg.norm(np.dot(A_val, w) - y_val)**2 / A_val.shape[0]
            self.err[it, 3] = np.linalg.norm(np.dot(A_test, w) - y_test) ** 2 / A_test.shape[0]
        best_lamb = np.argmin(self.err[:, 2])
        w = np.dot(np.dot(np.linalg.inv(np.dot(A_train.T, A_train) + best_lamb * I), A_train.T), y_train)
        print("MSE of Train")
        print(self.err[-1, 1])
        print("MSE of test")
        print(self.err[-1, 3])
        print("MSE of val")
        print(self.err[-1, 2])
        self.sol = w
        err = self.err
        print("best lambda is :", best_lamb)
        plt.figure()
        plt.plot(err[:, 0], err[:, 1], label='train')
        plt.plot(err[:, 0], err[:, 2], label='val')
        plt.xlabel('lambda')
        plt.ylabel('Mean Square Error')
        plt.legend()
        plt.title('Ridge error respect to lambda')
        plt.margins(0.01, 0.1)
        plt.xlim(xmin = 0, xmax = 300)
        plt.grid()
        plt.show()

class SolveGrad(SolveMinProb):
    def run(self, A_train, y_train, A_val, y_val, A_test, y_test, gamma = 1e-3, Nit = 100): #we need to specify the params
        self.err = np.zeros((Nit,4), dtype = float)
        '''
        :param gamma: learning coefficient. It's better to start 
        with small value of gamma and gradually manually increase it, 
        otherwise the algorithm could not converge. The correct value of 
        gamma depends on the specific func
        '''
        np.random.seed(3)
        w = np.random.rand(self.Nf, 1)
        for it in range(Nit):
            grad = 2 * np.dot(A_train.T,(np.dot(A_train,w)-y_train))
            w = w - gamma*grad
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm(y_train - np.dot(A_train, w)) ** 2 / A_train.shape[0]
            self.err[it, 2] = np.linalg.norm(y_val - np.dot(A_val, w)) ** 2 / A_val.shape[0]
            self.err[it, 3] = np.linalg.norm(y_test - np.dot(A_test, w)) ** 2 / A_test.shape[0]
        print("MSE of Train")
        print(self.err[-1, 1])
        print("MSE of test")
        print(self.err[-1, 3])
        print("MSE of val")
        print(self.err[-1, 2])
        self.sol = w
        self.min = self.err[it,1]

class SolveStochGrad(SolveMinProb):
    def run(self, A_train, y_train, A_val, y_val, A_test, y_test, gamma = 1e-3, Nit = 100):
        self.err = np.zeros((Nit, 4), dtype=float)
        Nf=A_train.shape[1]
        Np=A_train.shape[0]
        np.random.seed(3)
        w = np.random.rand(self.Nf, 1)
        row = np.zeros((1,Nf), dtype = float)
        for it in range(Nit):
            for i in range(Np):
                for j in range(Nf):
                    row[0,j] = A_train[i,j]
                grad = 2*row.T* (np.dot(row, w)-y_train[i])
                w = w-gamma*grad
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm(y_train - np.dot(A_train, w))**2/ A_train.shape[0]  # MSE_train
            self.err[it, 2] = np.linalg.norm(y_val - np.dot(A_val, w))**2/ A_val.shape[0]  # MSE_val
            self.err[it, 3] = np.linalg.norm(y_test - np.dot(A_test, w))**2/ A_test.shape[0]  # MSE_val
        print("MSE of Train")
        print(self.err[-1, 1])
        print("MSE of test")
        print(self.err[-1, 3])
        print("MSE of val")
        print(self.err[-1, 2])
        self.sol = w
        self.min = self.err[it, 1]

class SteepestDec(SolveMinProb):
    def run(self, A_train, y_train, A_val, y_val, A_test, y_test, gamma = 1e-3, Nit = 100):
        self.err = np.zeros((Nit,4), dtype = float)
        np.random.seed(3)
        w = np.random.rand(self.Nf, 1)
        '''
        :param gamma: the learning coefficient; it has to be optimized. 
        It's no more settled manually as in the gradient algorithm
        '''
        for it in range(Nit):
            grad = 2*np.dot(A_train.T, (np.dot(A_train, w)-y_train))
            H = 2*np.dot(A_train.T, A_train)
            gamma = np.power(np.linalg.norm(grad),2) / np.dot(np.dot(grad.T,H), grad)
            w = w - gamma*grad
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm(y_train - np.dot(A_train, w)) ** 2 / A_train.shape[0]
            self.err[it, 2] = np.linalg.norm(y_val - np.dot(A_val, w)) ** 2 / A_val.shape[0]
            self.err[it, 3] = np.linalg.norm(y_test - np.dot(A_test, w)) ** 2 / A_test.shape[0]
        print("MSE of Train")
        print(self.err[-1, 1])
        print("MSE of test")
        print(self.err[-1, 3])
        print("MSE of val")
        print(self.err[-1, 2])
        self.sol = w
        self.min = self.err[it, 1]

class ConjGrad(SolveMinProb):
    def run(self, A_train, A_val, A_test, y_train, y_val, y_test):
        self.err = np.zeros((self.Nf, 4), dtype=float)
        Q = 2*np.dot(A_train.T, A_train)
        np.random.seed(3)
        w = np.zeros((self.Nf, 1), dtype = float)
        b = np.dot(A_train.T, y_train)
        grad = -b
        d = -grad
        for it in range(A_train.shape[1]):
            alpha = - (np.dot(d.T,grad)/np.dot(np.dot(d.T,Q),d))
            w = w + d*alpha
            grad = grad + alpha*np.dot(Q,d)
            beta = (np.dot(np.dot(grad.T,Q),d)/np.dot(np.dot(d.T,Q),d))
            d = -grad + d*beta
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm(y_train - np.dot(A_train, w)) ** 2 / A_train.shape[0]
            self.err[it, 2] = np.linalg.norm(y_val - np.dot(A_val, w)) ** 2 / A_val.shape[0]
            self.err[it, 3] = np.linalg.norm(y_test - np.dot(A_test, w)) ** 2 / A_test.shape[0]
        print("MSE of Train")
        print(self.err[-1, 1])
        print("MSE of test")
        print(self.err[-1, 3])
        print("MSE of val")
        print(self.err[-1, 2])
        self.sol = w