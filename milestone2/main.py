import time
from milestone2.helper_Functions import *
from milestone2.models import *

X_train, X_test, Y_train, Y_test = pre_processing()

# Logistic Regression
t0 = time.time()
logisticRegressionAccuracy=logisticRegression(X_train, X_test, Y_train, Y_test)
t1 = time.time()
time_Taken = (t1 - t0)
print('Time taken by Logistic Regression', time_Taken, 'seconds')

# SVM with Polynomial kernel
polyC=2
polyD=3
t0 = time.time()
SVMPolykernelAccuracy=SVMPolykernel(X_train, X_test, Y_train, Y_test, polyC, polyD)
t1 = time.time()
time_Taken = (t1 - t0)
print('Time taken by SVM with Polynomial kernel', time_Taken, 'seconds')

# Decision Tree
t0 = time.time()
decisionTreeAccuracy=decisionTree(X_train, X_test, Y_train, Y_test)
t1 = time.time()
time_Taken = (t1 - t0)
print('Time taken by Decision Tree', time_Taken, 'seconds')

# SVM with Gaussian(RBF) kernel
rbfC=3
rbfG=3.1
t0 = time.time()
SVMRBFkernelAccuracy=SVMRBFkernel(X_train, X_test, Y_train, Y_test, rbfC, rbfG)
t1 = time.time()
time_Taken = (t1 - t0)
print('Time taken by SVM with Polynomial kernel', time_Taken, 'seconds')