import time
from sklearn.metrics import accuracy_score
from sklearn import svm
from helper_Functions import *

X_train, X_test, Y_train, Y_test = pre_processing()

t0 = time.time()
C = 3
rbf_svc = svm.SVC(kernel='rbf', gamma=0.8, C=C).fit(X_train, Y_train)
y_pred = rbf_svc.predict(X_test)
t1 = time.time()

time_Taken = (t1 - t0)
accuracy = accuracy_score(y_pred, Y_test)
print('Accuracy ', accuracy)
print('Time Taken ', time_Taken, 'seconds')
# 0.8408512916610895 C=0.1
# 0.9267835631106947 C=0.8
# 0.9313344933743809 C=3 48.11826205253601 seconds
# highest accuracy because:
# n=5 m=30,301 => n small m intermediate => use SVM with gaussian kernel
