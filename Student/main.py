import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))

data = pd.read_csv('student-mat.csv', sep=';')
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict = 'G3'

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

"""best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)

    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > best:
        best = acc
        # save model
        with open('studentmodel.pickle', 'wb') as f:
            pickle.dump(linear, f)"""

# open model
pickle_in = open('studentmodel.pickle', 'rb')
linear = pickle.load(pickle_in)

print('CO: {}'.format(linear.coef_))
print('Inter: {}'.format(linear.intercept_))

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print("Predicted: {} - Data: {} - Actual: {}".format(predictions[x], x_test[x], y_test[x]))

# Plotting scatter plot
predict = 'G1'
style.use('ggplot')
pyplot.scatter(data[predict],data['G3'])
pyplot.xlabel(predict)
pyplot.xlabel('Final Grade')
pyplot.show()