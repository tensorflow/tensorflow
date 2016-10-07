#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn import datasets, cross_validation, metrics
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn import monitors

# Load dataset

digits = datasets.load_digits()
X = digits.images
y = digits.target

# Split it into train / test subsets

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
                                                                     test_size=0.2,
                                                                     random_state=42)

# Split X_train again to create validation data

X_train, X_val, y_train, y_val = cross_validation.train_test_split(X_train,
                                                                   y_train,
                                                                   test_size=0.2,
                                                                   random_state=42)

# TensorFlow model using Scikit Flow ops


def conv_model(X, y):
  X = tf.expand_dims(X, 3)
  features = tf.reduce_max(tf.contrib.layers.conv2d(X, 12, [3, 3]), [1, 2])
  features = tf.reshape(features, [-1, 12])
  return learn.models.logistic_regression(features, y)

val_monitor = monitors.ValidationMonitor(X_val, y_val, every_n_steps=50)
# Create a classifier, train and predict.
classifier = learn.TensorFlowEstimator(model_fn=conv_model, n_classes=10,
                                        steps=1000, learning_rate=0.05,
                                        batch_size=128)
classifier.fit(X_train, y_train, monitors=[val_monitor])
score = metrics.accuracy_score(y_test, classifier.predict(X_test))
print('Test Accuracy: {0:f}'.format(score))
