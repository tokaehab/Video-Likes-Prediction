import time
from sklearn.metrics import accuracy_score
from sklearn import svm
from helper_Functions import *

X_train, X_test, Y_train, Y_test = pre_processing()

t0 = time.time()
clf = svm.SVC(C=2, kernel='poly', degree=3).fit(X_train, Y_train)
y_pred = clf.predict(X_test)
t1 = time.time()

time_Taken = (t1 - t0)
accuracy = accuracy_score(y_pred, Y_test)

print('Accuracy ', accuracy)
print('Time Taken ', time_Taken, 'seconds')
# 0.7126221389372239 C = 0.5
# 0.7312274126622942 C = 0.9
# 0.7499665372774729 C = 2
