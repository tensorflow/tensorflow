import random
from sklearn import datasets, cross_validation, metrics
import tensorflow as tf

import skflow

random.seed(42)

# Load dataset and split it into train / test subsets.

digits = datasets.load_digits()
X = digits.images
y = digits.target

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
    test_size=0.2, random_state=42)

# TensorFlow model using Scikit Flow ops

def conv_model(X, y):
    X = tf.expand_dims(X, 3)
    features = tf.reduce_max(skflow.ops.conv2d(X, 12, [3, 3]), [1, 2])
    features = tf.reshape(features, [-1, 12])
    return skflow.models.logistic_regression(features, y)

# Create a classifier, train and predict.
classifier = skflow.TensorFlowEstimator(model_fn=conv_model, n_classes=10,
                                        steps=500, learning_rate=0.05,
                                        batch_size=128)
classifier.fit(X_train, y_train)
score = metrics.accuracy_score(classifier.predict(X_test), y_test)
print('Accuracy: %f' % score)

