from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def logisticRegression(X_train, X_test, Y_train, Y_test):
    logreg = LogisticRegression().fit(X_train, Y_train)
    y_pred = logreg.predict(X_test)
    accuracy = accuracy_score(y_pred, Y_test)

    return accuracy
    # 0.8201044036942846 0.7477085590362549 seconds


def SVMPolykernel(X_train, X_test, Y_train, Y_test, c=2, d=3):
    clf = svm.SVC(C=c, kernel='poly', degree=d).fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_pred, Y_test)

    return accuracy
    # 0.7126221389372239 C = 0.5 degree=3 31.832858085632324 seconds
    # 0.7312274126622942 C = 0.9 degree=3 32.01036095619202 seconds
    # 0.7499665372774729 C = 2 degree=3 31.3341646194458 seconds


def decisionTree(X_train, X_test, Y_train, Y_test):
    clf = DecisionTreeClassifier().fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_pred, Y_test)

    return accuracy
    # 0.9242403961986347 0.09275078773498535 seconds


def SVMRBFkernel(X_train, X_test, Y_train, Y_test, c=3, g=3.1):
    rbf_svc = svm.SVC(kernel='rbf', gamma=g, C=c).fit(X_train, Y_train)
    y_pred = rbf_svc.predict(X_test)
    accuracy = accuracy_score(y_pred, Y_test)

    return accuracy
    # 0.8179627894525499 C=0.1 gamma=0.8 80.69420289993286 seconds
    # 0.904832017132914 C=0.8 gamma=0.8 61.1414692401886 seconds
    # 0.9117922634185517 C=1 gamma=0.8 52.491631507873535 seconds
    # 0.9313344933743809 C=3 gamma=0.8 48.11826205253601 seconds
    # 0.9338776602864409 C=3 gamma=1 59.5048463344574 seconds
    # 0.9376254852094766 C=3 gamma=2 73.03767776489258 seconds
    # 0.9404363539017534 C=3 gamma=3.1 114.8321213722229 seconds
    # 0.9399009503413198 C=3 gamma=3.2 83.3281397819519 seconds
    # highest accuracy because:
    # n=5 m=30,301 => n small m intermediate => use SVM with gaussian kernel