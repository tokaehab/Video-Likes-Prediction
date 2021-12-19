from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from helper_Functions import *

data, X = pre_processing()
# X = data[['views', 'comment_count']].iloc[:, :]
X = feature_scaling(X, 0, 100)
Y = data[['VideoPopularity']].iloc[:, :]
Y = feature_scaling(Y, 0, 100)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=1)

C = 3
rbf_svc = svm.SVC(kernel='rbf', gamma=0.8, C=C).fit(X, Y)
y_pred = rbf_svc.predict(X_test)

accuracy = accuracy_score(y_pred, y_test)
print(accuracy)
# 0.8408512916610895 C=0.1
# 0.9267835631106947 C=0.8
# 0.9440503279346808 C=2
# 0.9490028108686923 C=3
# highest accuracy because:
# n=5 m=30,301 => n small m intermediate => use SVM with gaussian kernel
