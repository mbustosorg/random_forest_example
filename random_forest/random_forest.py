import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt

""" XOR Model """
X = [[-1, -1], [-1, 1], [1, -1], [0, 0]]
y = [0, 1, 1, 0]

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)
print(clf.predict([[-1, -1], [-1, 1], [1, -1], [0, 0]]))

""" Random model, 10 inputs with 4 classes """
if True:
    X, y = make_classification(n_samples=5000, n_features=10,
                               n_informative=5, n_redundant=0,
                               n_classes=4,
                               random_state=0, shuffle=False)
    np.savetxt("x.csv", X, delimiter=",")
    np.savetxt("y.csv", y, delimiter=",")
    with open('random.npy', 'wb') as random_data:
        np.save(random_data, X)
        np.save(random_data, y)
else:
    with open('random.npy', 'rb') as random_data:
        X = np.load(random_data)
        y = np.load(random_data)
clf = RandomForestClassifier(random_state=0)
clf.fit(X, y)
pred_y = clf.predict(X)
print(confusion_matrix(y, pred_y))
print((y - pred_y).sum())

""" Write out ONNX """
initial_type = [('float_input', FloatTensorType([None, 10]))]
onx = convert_sklearn(clf, initial_types=initial_type)
with open("random.onnx", "wb") as f:
    f.write(onx.SerializeToString())

""" Predict with ONNX """
sess = rt.InferenceSession("random.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run([label_name], {input_name: X.astype(numpy.float32)})[0]