import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model, preprocessing
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import os
import matplotlib.pyplot as pyplot
from matplotlib import style

os.chdir(os.path.dirname(os.path.realpath(__file__)))

data = pd.read_csv('car.data')

# Making data strings into ints
le = preprocessing.LabelEncoder()

# List for each column
buying = le.fit_transform(list(data['buying']))
maint = le.fit_transform(list(data['maint']))
door = le.fit_transform(list(data['door']))
persons = le.fit_transform(list(data['persons']))
lug_boot = le.fit_transform(list(data['lug_boot']))
safety = le.fit_transform(list(data['safety']))
cls = le.fit_transform(list(data['class']))
# print(buying)

predict = 'class'

# Feature
X = list(zip(buying, maint, door, persons, lug_boot, safety))
# Label
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# print(x_train, y_train)
model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

predictions = model.predict(x_test)
names = ['unacc', 'acc', 'good', 'vgood']

for x in range(len(predictions)):
    if predictions[x] != y_test[x]:
        print("Predicted: {} - Data: {} - Actual: {} WRONG".format(names[predictions[x]], x_test[x], names[y_test[x]]))
    else:
        print("Predicted: {} - Data: {} - Actual: {}".format(names[predictions[x]], x_test[x], names[y_test[x]]))
    n = model.kneighbors([x_test[x]], 9, True)
    print("N: {}".format(n))

# Plotting scatter plot
predict = 'buying'
style.use('ggplot')
pyplot.scatter(data[predict],data['safety'])
pyplot.xlabel(predict)
pyplot.xlabel('Safety')
pyplot.show()