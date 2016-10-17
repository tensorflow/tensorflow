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
"""
This example uses the same data as one here:
  http://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression_multioutput.html

Instead of DecisionTree a 2-layer Deep Neural Network with RELU activations is used.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import tensorflow as tf

from tensorflow.contrib import learn

# Create random dataset.
rng = np.random.RandomState(1)
X = np.sort(200 * rng.rand(100, 1) - 100, axis=0)
y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T

# Fit regression DNN models.
regressors = []
options = [[2], [10, 10], [20, 20]]
for hidden_units in options:
  def tanh_dnn(X, y):
    dnn = lambda inputs, num_outputs, scope: tf.contrib.layers.legacy_fully_connected(
        inputs, num_outputs, weight_init=None, activation_fn=learn.tf.tanh)
    features = tf.contrib.layers.stack(X, dnn, hidden_units)
    return learn.models.linear_regression(features, y)

  regressor = learn.TensorFlowEstimator(model_fn=tanh_dnn, n_classes=0,
      steps=500, learning_rate=0.1, batch_size=100)
  regressor.fit(X, y)
  score = mean_squared_error(regressor.predict(X), y)
  print("Mean Squared Error for {0}: {1:f}".format(str(hidden_units), score))
  regressors.append(regressor)

# Predict on new random Xs.
X_test = np.arange(-100.0, 100.0, 0.1)[:, np.newaxis]
y_1 = regressors[0].predict(X_test)
y_2 = regressors[1].predict(X_test)
y_3 = regressors[2].predict(X_test)

# Plot the results
plt.figure()
plt.scatter(y[:, 0], y[:, 1], c="k", label="data")
plt.scatter(y_1[:, 0], y_1[:, 1], c="g",
    label="hidden_units{}".format(str(options[0])))
plt.scatter(y_2[:, 0], y_2[:, 1], c="r",
    label="hidden_units{}".format(str(options[1])))
plt.scatter(y_3[:, 0], y_3[:, 1], c="b",
    label="hidden_units{}".format(str(options[2])))
plt.xlim([-6, 6])
plt.ylim([-6, 6])
plt.xlabel("data")
plt.ylabel("target")
plt.title("Multi-output DNN Regression")
plt.legend()
plt.show()
