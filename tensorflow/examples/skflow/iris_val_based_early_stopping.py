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

from sklearn import datasets, metrics
from sklearn.cross_validation import train_test_split

from tensorflow.contrib import skflow


iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data,
                                                    iris.target,
                                                    test_size=0.2,
                                                    random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size=0.2, random_state=42)
val_monitor = skflow.monitors.ValidationMonitor(X_val, y_val,
                                                early_stopping_rounds=200,
                                                n_classes=3)

# classifier with early stopping on training data
classifier1 = skflow.TensorFlowDNNClassifier(hidden_units=[10, 20, 10],
                                             n_classes=3, steps=2000)
classifier1.fit(X_train, y_train)
score1 = metrics.accuracy_score(y_test, classifier1.predict(X_test))

# classifier with early stopping on validation data
classifier2 = skflow.TensorFlowDNNClassifier(hidden_units=[10, 20, 10],
                                             n_classes=3, steps=2000)
classifier2.fit(X_train, y_train, val_monitor)
score2 = metrics.accuracy_score(y_test, classifier2.predict(X_test))

# in many applications, the score is improved by using early stopping on val data
print(score2 > score1)
