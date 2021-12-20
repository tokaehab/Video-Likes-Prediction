import time
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from helper_Functions import *

X_train, X_test, Y_train, Y_test = pre_processing()

t0 = time.time()
clf = DecisionTreeClassifier().fit(X_train, Y_train)
y_pred = clf.predict(X_test)
t1 = time.time()

time_Taken = (t1 - t0)
accuracy = accuracy_score(y_pred, Y_test)

print('Accuracy ', accuracy)
print('Time Taken ', time_Taken, 'seconds')
# 0.9242403961986347
