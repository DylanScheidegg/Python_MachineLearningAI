import sklearn
from sklearn import datasets
from sklearn import svm
import os
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

os.chdir(os.path.dirname(os.path.realpath(__file__)))

cancer = datasets.load_breast_cancer()
# print(cancer.feature_names)
# print(cancer.target_names)

X = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

clf = svm.SVC(kernel='linear', C=2)
# clf = svm.SVC(kernel='poly', degree=2)
clf.fit(x_train, y_train)
predictions = clf.predict(x_test)
acc = metrics.accuracy_score(y_test, predictions)
print(acc)

"""model = KNeighborsClassifier(n_neighbors=9)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)"""
