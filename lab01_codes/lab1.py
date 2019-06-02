from min_class import *
import matplotlib.pyplot as plt
import pandas as pd
import csv
import numpy as np
import random
import math

def data_initialization(name):
    np.random.seed(3)
    # retrieving data
    x = pd.read_csv(name, header=None)
    # x.info()
    header_names = list(x.iloc[0, :])

    x = pd.read_csv(name)
    x = x.sample(frac=1)  # shuffle: frac=1 means return all rows in random order

    # remove the first 4 columns (0, 1, 2, 3)
    x = x.iloc[:, 4:22]  # iloc = integer-location
    header_names = header_names[4:22]
    data = x.values

    # normalizing the data
    MEAN = data.mean(axis=0)  # axis=0 stands for X and axis=1 stands for columns
    ST_DEV = data.std(axis=0)
    data_norm = (data - MEAN)/ST_DEV

    if int(np.sum(np.mean(data_norm, axis=0))) == 0 and np.sum(np.var(data_norm, axis=0)) == data_norm.shape[1]:
        print('data {} successfully normalized. Columns mean = 0 and Columns variance = 1.'.format(name))
    else:
        print('data {} normalization was unsuccessful.quitting.'.format(name))
        quit()
    return data_norm, data, MEAN, ST_DEV, header_names


if __name__ == "__main__":
    np.random.seed(3)

    data_norm, data, MEAN, ST_DEV, header_names = data_initialization("parkinsons_updrs.csv")

    training_end_row = math.floor((len(data_norm)*0.5))
    validation_end_row = math.floor((len(data_norm)*0.75))

    data_train_norm = data_norm[0:training_end_row, :].copy()
    data_test_norm = data_norm[training_end_row+1:validation_end_row, :].copy()
    data_val_norm = data_norm[validation_end_row+1:-1, :].copy()

    data_train = data[0:training_end_row, :].copy()
    data_test = data[training_end_row+1:validation_end_row, :].copy()
    data_val = data[validation_end_row+1:-1, :].copy()

    # F0 is the feature that we want to estimate from the other features
    F0 = 1
    y_train = data_train_norm[:, F0]
    y_test = data_test_norm[:, F0]
    y_val = data_val_norm[:, F0]

    # making sure that all data is column vector
    y_train = y_train.reshape(y_train.shape[0], 1)  # xxx.shape[0] returns the index size of xxx (nbr of rows)
    y_test = y_test.reshape(y_test.shape[0], 1)
    y_val = y_val.reshape(y_val.shape[0], 1)

    Np = data.shape[0]  # number of columns
    # # if I normalize the data it's no more necessary to add 1 column of all ones to the data:
    # new_data_ones = np.c_[data, np.ones(Np)]
    # y_train_ones = new_data_ones[:, F0]
    # y_test_ones = new_data_ones[:, F0]

    X_train = np.delete(data_train_norm, F0, axis=1)  # deleting the column F0
    X_test = np.delete(data_test_norm, F0, axis=1)
    X_val = np.delete(data_val_norm, F0, axis=1)
    y_name = header_names[F0]
    del header_names[F0]
    logx = 0
    logy = 0

    # m = SolveLLS(y_train, X_train)
    # m.run(X_train, y_train, X_test, y_test, X_val, y_val)
    # m.print_result('LLS')
    #
    # fig = plt.figure()
    #
    # m.plot_w('Regressors weight obtained from LLS\nRegressand: '+y_name, header_str=header_names)
    # m.plot_y(X_test, X_train, y_test, y_train, MEAN[F0], ST_DEV[F0], 'y versus ŷ', plt_handle=fig)
    # m.plot_hist(X_train, X_test, y_train, y_test, "Estimation Error Histogram", plt_handle=fig)
    # fig.suptitle('Regressand: '+y_name, fontsize=10, verticalalignment='top')
    # fig.tight_layout(rect=[0, 0, 1, 0.60])
    #
    # fig.show()


    # cong = ConjGrad(y_train, X_train)
    # cong.run(X_train, X_val, X_test, y_train, y_val, y_test)
    # cong.print_result('ConjGrad.')
    # fig = plt.figure()
    # cong.plot_y(X_test, X_train, y_test, y_train, MEAN[F0], ST_DEV[F0], 'y versus ŷ', plt_handle=fig)
    # cong.plot_hist(X_train, X_test, y_train, y_test, "Estimation Error Histogram", plt_handle=fig)
    # fig.suptitle('Method: Conjugate Gradient. Regressand: '+y_name, fontsize=10, verticalalignment='top')
    # fig.tight_layout(rect=[0, 0, 1, 0.60])
    # cong.plot_w('Regressors weight obtained from CG\nRegressand: '+y_name, header_str=header_names)
    # cong.plot_err('Error Rate', logy, logx)


    # Nit = 200
    # gamma = 1e-5
    # # # #'''
    # # # #  It’s important with this algorithm to set appropriate values of Nit and γ : in order to choose
    # # # #  right values, the program has to be run several times. Moreover, if γ is too small,
    # # # #  it could take a lot of time to the program to find the optimum value of w; if γ is too large,
    # # # #  the algorithm could not converge.
    # # # # '''
    # g = SolveGrad(y_train, X_train)
    # g.run(X_train, y_train, X_val, y_val, X_test, y_test, gamma, Nit)
    # g.print_result('Gradient algo.')
    # g.plot_err('Error Rate', logy, logx) # find stop point: over fitting?
    # fig = plt.figure()
    # g.plot_y(X_test, X_train, y_test, y_train, MEAN[F0], ST_DEV[F0], 'y versus ŷ', plt_handle=fig)
    # g.plot_hist(X_train, X_test, y_train, y_test, "Estimation Error Histogram", plt_handle=fig)
    # fig.suptitle('Method: Gradient Algorithm. Regressand: '+y_name, fontsize=10, verticalalignment='top')
    # fig.tight_layout(rect=[0, 0, 1, 0.60])
    # g.plot_w('Regressors weight obtained from Gradient Algorithm\nRegressand: '+y_name, header_str=header_names)


    # Nit = 600
    # gamma = 1e-6
    # st = SolveStochGrad(y_train, X_train)
    # st.run( X_train, y_train, X_val, y_val, X_test, y_test, gamma, Nit)
    # st.print_result("Sthocastic Gradient")
    # fig = plt.figure()
    # st.plot_y(X_test, X_train, y_test, y_train, MEAN[F0], ST_DEV[F0], 'y versus ŷ', plt_handle=fig)
    # st.plot_hist(X_train, X_test, y_train, y_test, "Estimation Error Histogram", plt_handle=fig)
    # fig.suptitle('Method: Stochastic Gradient. Regressand: '+y_name, fontsize=10, verticalalignment='top')
    # fig.tight_layout(rect=[0, 0, 1, 0.60])
    #
    # st.plot_err('Error Rate', logy, logx)
    # st.plot_w('Regressors weight obtained from SG\nRegressand: '+y_name, header_str=header_names)


    # Nit = 300
    # gamma = 1e-3
    # s = SteepestDec(y_train, X_train)
    # s.run(X_train, y_train, X_val, y_val, X_test, y_test, gamma, Nit)
    # s.print_result("Steepest decent")
    #
    # fig = plt.figure()
    # s.plot_y(X_test, X_train, y_test, y_train, MEAN[F0], ST_DEV[F0], 'y versus ŷ', plt_handle=fig)
    # s.plot_hist(X_train, X_test, y_train, y_test, "Estimation Error Histogram", plt_handle=fig)
    # fig.suptitle('Method: Steepest Descent. Regressand: '+y_name, fontsize=10, verticalalignment='top')
    # fig.tight_layout(rect=[0, 0, 1, 0.60])
    # fig.show()
    # s.plot_err('Error Rate', logy, logx)
    # s.plot_w('Regressors weight obtained from SD\nRegressand: '+y_name, header_str=header_names)


    #Ridge regression. when lamba = 0 , we are again in the case of LLS, VERIFIED.
    sr = SolveRidge(y_train, X_train)
    sr.run(X_train, X_val, X_test, y_train, y_val, y_test)
    fig = plt.figure()

    sr.plot_y(X_test, X_train, y_test, y_train, MEAN[F0], ST_DEV[F0], 'y versus ŷ', plt_handle=fig)
    sr.plot_hist(X_train, X_test, y_train, y_test, "Estimation Error Histogram", plt_handle=fig)
    fig.suptitle('Method: Ridge Regression. Regressand: '+y_name, fontsize=10, verticalalignment='top')
    fig.tight_layout(rect=[0, 0, 1, 0.60])
    fig.show()
    sr.plot_w('Regressors weight obtained from RR\nRegressand: '+y_name, header_str=header_names)
    sr.print_result('RR algo.')
