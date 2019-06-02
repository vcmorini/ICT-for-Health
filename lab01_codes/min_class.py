import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error


class SolveMinProb:
    def __init__(self, y=np.ones((3,1)), A=np.eye(3)):
        np.random.seed(3)
        self.matr = A
        self.Np = y.shape[0]  # number of columns
        self.Nf = A.shape[1]  # number of rows
        self.vect = y
        self.sol = np.zeros((self.Nf, 1), dtype=float)
        return

    def plot_w(self, title='Weights', header_str=''):
        w = self.sol
        n = np.arange(self.Nf)  # Return evenly spaced values within a given interval.
        plt.figure()
        plt.stem(n, w)  # looks like graph bar
        # max = np.max(w)
        # min = np.min(w)
        plt.xlabel('regressors')
        plt.ylabel('w(weight of each regressor)')

        plt.ylim(bottom=np.min(w)*1.1, top=np.max(w)*1.1)
        plt.title(title)
        plt.xticks(n, header_str, rotation=90)
        plt.grid()
        plt.show()
        return

    def plot_y(self, A_test, A_train, y_test, y_train,  mean, st_dev, title, plt_handle=plt.figure()):
        '''
        :param self is the obj itself
        :param A_test: matrix of test data set
        :param A_train: matrix of train data set. This is used to find w
        :param y_test: column taken from the matrix A_test, de-normalized
        :param y_train: column taken from the matrix A_train, de-normalized
        :param mean: row vector: each element is the mean calculated for each column of the  matrix containing all data
        :param st_dev: vector of mean calculated for each column of the  matrix containing all data
        :param title of the graph
        :return: returns the plot of y and y_hat
        '''

        w = self.sol
        '''De-normalization of the vector y'''

        y_hat_train = np.dot(A_train, w)*st_dev + mean
        y_hat_test = np.dot(A_test, w)*st_dev + mean
        y_train = y_train*st_dev + mean
        y_test = y_test*st_dev + mean

        axis0 = plt_handle.add_subplot(2, 2, 1)
        slope, intercept, r_value, p_value, std_err = stats.mstats.linregress(y_hat_train,y_train)
        line = slope*y_hat_train+intercept
        # axis0.set_title(title)
        axis0.plot(y_hat_train, line, color='black')
        axis0.scatter(y_hat_train, y_train, s=1)  # parameter 's' is the area of the scatter point in the graph
        axis0.set_xlabel('y_hat_train')
        axis0.set_ylabel('y_train')
        axis0.set_title('a) '+title+'\ntrain dataset', fontdict={'fontsize': 9}, loc='left')
        axis0.grid()

        axis1 = plt_handle.add_subplot(2, 2, 3)
        slope, intercept, r_value, p_value, std_err = stats.mstats.linregress(y_hat_test, y_test)
        line = slope * y_hat_test + intercept
        axis1.plot(y_hat_test, line, color='black')
        axis1.scatter(y_hat_test, y_test, s=1, color='orange')
        axis1.set_xlabel('y_hat_test')
        axis1.set_ylabel('y_test')
        axis1.set_title('b) '+title+'\ntest dataset', fontdict={'fontsize': 9}, loc='left')
        axis1.grid()

        all = np.concatenate((y_hat_train, y_hat_test, y_train, y_test), axis=None)
        axis1.set_xlim(left=np.amin(all), right=np.amax(all))
        axis1.set_ylim(bottom=np.amin(all), top=np.amax(all))
        axis0.set_xlim(left=np.amin(all), right=np.amax(all))
        axis0.set_ylim(bottom=np.amin(all), top=np.amax(all))
        return

    def plot_hist(self, A_train, A_test, y_train, y_test, title, plt_handle=plt.figure()):
        print(title, '')
        '''
        This method is used to plot the histograms of y_hat_train-y_hat and y_hat_test-y_test
        '''
        w = self.sol
        y_hat_train = np.dot(A_train, w)
        y_hat_test = np.dot(A_test, w)
        error_test = y_test - y_hat_test
        error_train = y_train - y_hat_train

        axis0 = plt_handle.add_subplot(2, 2, 4)
        n0, bins0, patches0 = axis0.hist(error_test, bins=50, color='orange')
        axis0.set_xlabel('ŷ_test-y_test')
        axis0.set_ylabel('number occurencies')
        axis0.set_title('d) '+title+'\ntest dataset', fontdict={'fontsize': 9}, loc='left')
        axis0.grid()
        # axis0.savefig(title+'.pdf', bbox_inches='tight')

        axis1 = plt_handle.add_subplot(2, 2, 2)
        n1,bins1,patches1= axis1.hist(error_train, bins=50)  # arguments are passed to np.histogram
        axis1.set_xlabel('ŷ_train-y_train')
        axis1.set_ylabel('number occurences')
        axis1.set_title('c) '+title+'\ntrain dataset', fontdict={'fontsize': 9}, loc='left')
        axis1.grid()

        n = np.concatenate((n0, n1), axis=None)
        bins = np.concatenate((bins0, bins1), axis=None)

        axis1.set_xlim(left=np.amin(bins), right=np.amax(bins))
        axis1.set_ylim(bottom=np.amin(n), top=np.amax(n))
        axis0.set_xlim(left=np.amin(bins), right=np.amax(bins))
        axis0.set_ylim(bottom=np.amin(n), top=np.amax(n))
        return

    def print_result(self, title):
        print(title, ' ')
        print('the optimum weight vector is: ')
        print(self.sol)
        return

    def plot_err(self,title = 'Mean Square error', logy = 0, logx = 0):
        ''' this method plots the Mean Square Error'''
        err = self.err
        plt.figure()
        if (logy == 0) & (logx == 0):
            plt.plot(err[:,0], err[:,1], label='train')
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
        plt.xlabel('Training Iterations')
        plt.ylabel('Mean Square Error')
        plt.title(title)
        plt.margins(0.01, 0.1)
        plt.legend()
        plt.grid()
        plt.xlim(left = 0, right = 100)
        plt.show()


class SolveLLS (SolveMinProb):
    def run(self, A_train, y_train, A_test, y_test, A_val, y_val):
        np.random.seed(3)
        w = np.dot(np.linalg.pinv(A_train), y_train)  # Compute the (Moore-Penrose) pseudo-inverse of a matrix
        self.sol = w
        self.min = np.linalg.norm(np.dot(A_train, w) - y_train)
        self.MSE_train = np.linalg.norm(np.dot(A_train, w) - y_train)**2/A_train.shape[0]
        # self.MSE_train = mean_squared_error(np.dot(A_train, w), y_train)
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
        best_lamb = np.argmin(self.err[:, 3])  # Returns the indices of the minimum values along an axis.
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
        plt.plot(err[:, 0], err[:, 3], label='test')
        plt.plot(err[:, 0], err[:, 2], label='val')
        plt.xlabel('lambda')
        plt.ylabel('Mean Square Error')
        plt.legend()
        plt.title('Error Rate x lambda')
        plt.margins(0.01, 0.1)
        plt.xlim(left=0, right=300)
        plt.grid()
        plt.show()


class SolveGrad(SolveMinProb):
    def run(self, A_train, y_train, A_val, y_val, A_test, y_test, gamma = 1e-3, Nit = 100):  # we need to specify the params
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
            grad = 2 * np.dot(A_train.T, (np.dot(A_train, w)-y_train))
            w = w - gamma*grad
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm(y_train - np.dot(A_train, w)) ** 2 / A_train.shape[0]
            self.err[it, 2] = np.linalg.norm(y_val - np.dot(A_val, w)) ** 2 / A_val.shape[0]
            self.err[it, 3] = np.linalg.norm(y_test - np.dot(A_test, w)) ** 2 / A_test.shape[0]
        print("MSE of Train")
        print(self.err[-1, 1])  # '-1' refers to the last row, i.e. the last iteration 'it'
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
        np.random.seed(3)
        self.err = np.zeros((self.Nf, 4), dtype=float)
        Q = np.dot(A_train.T, A_train)  # because it is not symmetrical/Hermitian
        w = np.zeros((self.Nf, 1), dtype = float)
        b = np.dot(A_train.T, y_train)  # because it is not symmetrical/Hermitian
        grad = -b
        d = -grad
        for it in range(A_train.shape[1]):
            alpha = - (np.dot(d.T, grad)/np.dot(np.dot(d.T, Q), d))
            w = w + d*alpha
            grad = grad + alpha*np.dot(Q,d)
            beta = (np.dot(np.dot(grad.T, Q), d)/np.dot(np.dot(d.T, Q),d))
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