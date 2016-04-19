#  Copyright 2015-present The Scikit Flow Authors. All Rights Reserved.
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
from sklearn import preprocessing

from tensorflow.contrib import skflow

# Load dataset
boston = datasets.load_boston()
X, y = boston.data, boston.target

# Split dataset into train / test
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
    test_size=0.2, random_state=42)

# Scale data (training set) to 0 mean and unit standard deviation.
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)

# Build 2 layer fully connected DNN with 10, 10 units respectively.
regressor = skflow.TensorFlowDNNRegressor(hidden_units=[10, 10],
    steps=5000, learning_rate=0.1, batch_size=1)

# Fit
regressor.fit(X_train, y_train)

# Predict and score
score = metrics.mean_squared_error(regressor.predict(scaler.fit_transform(X_test)), y_test)

print('MSE: {0:f}'.format(score))
