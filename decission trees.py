import numpy as np
from sklearn import tree

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = [0, 0, 1, 1]

clf = tree.DecisionTreeClassifier(random_state=0)
clf.fit(X, y)
print(clf.score(X, y))
print(clf.predict([[2, 2]]))
