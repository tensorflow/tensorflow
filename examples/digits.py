#  Copyright 2015-present Scikit Flow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

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
print('Accuracy: {0:f}'.format(score))
