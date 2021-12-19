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

clf = svm.SVC(C=2, kernel='poly', degree=3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_pred, y_test)
# accuracy = np.mean(y_pred == y_test) # TODO: is there a difference?
print(accuracy)
# 0.7126221389372239 C = 0.5
# 0.7312274126622942 C = 0.9
# 0.7499665372774729 C = 2
