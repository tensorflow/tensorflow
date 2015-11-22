import random

from sklearn import datasets, metrics

import skflow

random.seed(42)

# Load dataset.
iris = datasets.load_iris()

# Build 3 layer DNN with 10, 20, 10 units respecitvely.
classifier = skflow.TensorFlowDNNClassifier(hidden_units=[10, 20, 10],
n_classes=3, steps=200)

# Fit and predict.
classifier.fit(iris.data, iris.target)
score = metrics.accuracy_score(classifier.predict(iris.data), iris.target)
print("Accuracy: %f" % score)

