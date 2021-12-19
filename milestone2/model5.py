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
lin_svc = svm.LinearSVC(C=C).fit(X, Y)
y_pred = lin_svc.predict(X_test)

accuracy = accuracy_score(y_pred, y_test)
print(accuracy)
# 0.6547985544103868 C=3
