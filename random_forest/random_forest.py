import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

""" XOR Model """
X = [[-1, -1], [-1, 1], [1, -1], [0, 0]]
y = [0, 1, 1, 0]

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)
print(clf.predict([[-1, -1], [-1, 1], [1, -1], [0, 0]]))

""" Random model, 10 inputs with 4 classes """
if True:
    X, y = make_classification(n_samples=1000, n_features=10,
                               n_informative=5, n_redundant=0,
                               n_classes=4,
                               random_state=0, shuffle=False)
    with open('random.npy', 'wb') as random_data:
        np.save(random_data, X)
        np.save(random_data, y)
else:
    with open('random.npy', 'rb') as random_data:
        X = np.load(random_data)
        y = np.load(random_data)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)
print(clf.predict(X))
print()
