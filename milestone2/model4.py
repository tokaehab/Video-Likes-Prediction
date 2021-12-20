import time
from sklearn.metrics import accuracy_score
from sklearn import svm
from helper_Functions import *

X_train, X_test, Y_train, Y_test = pre_processing()

t0 = time.time()
C = 3
rbf_svc = svm.SVC(kernel='rbf', gamma=3.1, C=C).fit(X_train, Y_train)
y_pred = rbf_svc.predict(X_test)
t1 = time.time()

time_Taken = (t1 - t0)
accuracy = accuracy_score(y_pred, Y_test)
print('Accuracy ', accuracy)
print('Time Taken ', time_Taken, 'seconds')
# 0.8179627894525499 C=0.1 gamma=0.8 80.69420289993286 seconds
# 0.904832017132914 C=0.8 gamma=0.8 61.1414692401886 seconds
# 0.9117922634185517 C=1 gamma=0.8 52.491631507873535 seconds
# 0.9313344933743809 C=3 gamma=0.8 48.11826205253601 seconds
# 0.9338776602864409 C=3 gamma=1 59.5048463344574 seconds
# 0.9376254852094766 C=3 gamma=2 73.03767776489258 seconds
# 0.9404363539017534 C=3 gamma=3.1 114.8321213722229 seconds
# highest accuracy because:
# n=5 m=30,301 => n small m intermediate => use SVM with gaussian kernel
