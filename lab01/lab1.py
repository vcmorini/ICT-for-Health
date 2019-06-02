from lab_01.min_class import *
import matplotlib.pyplot as plt
import pandas as pd
import csv
import numpy as np
import random


x = pd.read_csv("parkinsons_updrs.csv")
x.info()
# shuffle: frac=1 means return all rows in random order
x = x.sample(frac=1)

# remove the first 4 columns
x = x.iloc[:,4:22]
data = x.values

# normalizing the data
MEAN = data.mean(axis = 0)
ST_DEV = data.std(axis = 0)
data_norm = (data - data.mean(axis = 0))/ data.std(axis = 0)
# print("mean of data_norm: ")
# print(np.mean(data_norm, axis = 0)) # it should be 0 for each column
# print("variance of data_norm: ")
# print(np.var(data_norm, axis = 0)) # it should be 1 for each column

if __name__ == "__main__":

    data_train_norm = data_norm[:2937, :].copy()
    data_test_norm = data_norm[2938:4405, :].copy()
    data_val_norm = data_norm[4406:5874, :].copy()

    data_train = data[:2937, :].copy()
    data_test = data[2938:4405, :].copy()
    data_val = data[4406:5874, :].copy()
    # F0 is the feature that we want to estimate (regress) from the other features(regressand).
    F0 = 1
    y_train = data_train_norm[:, F0]
    y_test = data_test_norm[:, F0]
    y_val = data_val_norm[:, F0]

    y_train = y_train.reshape(y_train.shape[0],1)
    y_test = y_test.reshape(y_test.shape[0],1)
    y_val = y_val.reshape(y_val.shape[0],1)

    Np = data.shape[0]
    # # if I normalize the data it's no more necessary to add 1 column of all ones to the data:
    # new_data_ones = np.c_[data, np.ones(Np)]
    # y_train_ones = new_data_ones[:, F0]
    # y_test_ones = new_data_ones[:, F0]


    X_train = np.delete(data_train_norm, F0, 1)
    X_test = np.delete(data_test_norm, F0, 1)
    X_val = np.delete(data_val_norm, F0, 1)
    logx = 0
    logy = 0


    # m = SolveLLS(y_train, X_train)  # instatiate the object
    # m.run(X_train, y_train, X_test, y_test, X_val, y_val)
    # m.print_result('LLS')
    # m.plot_w('LLS')
    # m.plot_y(X_test, X_train, y_test, y_train, MEAN[F0], ST_DEV[F0], 'SLL y versus ŷ')
    # m.plot_hist(X_train, X_test, y_train, y_test, "LLS Hist, estimation error")

    #
    # cong = ConjGrad(y_train, X_train)
    # cong.run(X_train, X_val, X_test, y_train, y_val, y_test)
    # cong.print_result('ConjGrad.')
    # cong.plot_err('ConjGrad Algo Mean Square error', logy, logx)
    # cong.plot_y(X_test, X_train,y_test, y_train, MEAN[F0], ST_DEV[F0], 'ConjGrad y versus ŷ')
    # cong.plot_hist(X_train, X_test, y_train, y_test, "Conjugate Grad Hist, estimation error")
    # cong.plot_w('Conjugate  w')
    #
    # # Nit = 200
    # gamma = 1e-5
    # # #'''
    # # #  It’s important with this algorithm to set appropriate values of Nit and γ : in order to choose
    # # #  right values, the program has to be run several times. Moreover, if γ is too small,
    # # #  it could take a lot of time to the program to find the optimum value of w; if γ is too large,
    # # #  the algorithm could not converge.
    # # # '''
    # g = SolveGrad(y_train, X_train)
    # g.run(X_train, y_train, X_val, y_val, X_test, y_test, gamma, Nit)
    # g.print_result('Gradient algo.')
    # g.plot_err('Gradient algo Mean Square Error', logy, logx) # find stop point: over fitting?
    # g.plot_y(X_test, X_train, y_test, y_train, MEAN[F0], ST_DEV[F0], 'GradAlgo y versus ŷ ')
    # g.plot_hist(X_train, X_test, y_train, y_test, "GradAlgo Hist, estimation error")
    # g.plot_w("gradient")

    # Nit = 600
    # gamma = 1e-6
    # st = SolveStochGrad(y_train, X_train)
    # st.run( X_train, y_train, X_val, y_val, X_test, y_test, gamma, Nit)
    # st.print_result("Sthocastic Gradient")
    # st.plot_err("Stochastic Gradient Mean Square Error", logy, logx)
    # st.plot_y(X_test, X_train, y_test, y_train, MEAN[F0], ST_DEV[F0],  'StochGrad y versus ŷ')
    # st.plot_hist(X_train, X_test, y_train, y_test, "StochGrad Hist, estimation error")
    # st.plot_w('Stochastic gradient w')

    # Nit = 300
    # gamma = 1e-3
    # s = SteepestDec(y_train, X_train)
    # s.run(X_train, y_train, X_val, y_val, X_test, y_test, gamma, Nit)
    # s.print_result("Steepest decent")
    # s.plot_err('Steepest decent: square error', logy, logx)
    # s.plot_y(X_test, X_train, y_test, y_train, MEAN[F0], ST_DEV[F0],'SeepestDescent y versus ŷ')
    # s.plot_hist(X_train, X_test, y_train, y_test, "SeepestDescent Hist, estimation error")
    # s.plot_w('Steepest Descent, w')

    #Ridge regression. when lamba = 0 , we are again in the case of LLS, VERIFIED.
    sr = SolveRidge(y_train, X_train)
    sr.run(X_train, X_val, X_test, y_train, y_val, y_test)
    sr.plot_y(X_test, X_train, y_test, y_train, MEAN[F0], ST_DEV[F0], 'Ridge Regression y versus ŷ')
    sr.plot_hist(X_train, X_test, y_train, y_test, "Solve Ridge Hist, estimation error")
    sr.plot_w('Solve Ridge ')
    sr.print_result('RR algo.')
