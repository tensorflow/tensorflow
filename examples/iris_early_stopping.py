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

import tensorflow as tf
from tensorflow.python.platform import googletest

from sklearn import datasets, metrics
from sklearn.cross_validation import train_test_split

import skflow


iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data,
                                                    iris.target,
                                                    test_size=0.2,
                                                    random_state=42)

# classifier without early stopping - overfitting
classifier1 = skflow.TensorFlowDNNClassifier(hidden_units=[10, 20, 10],
                                            n_classes=3, steps=800)
classifier1.fit(X_train, y_train)
score1 = metrics.accuracy_score(y_test, classifier1.predict(X_test))

# classifier with early stopping - improved accuracy on testing set
classifier2 = skflow.TensorFlowDNNClassifier(hidden_units=[10, 20, 10],
                                            n_classes=3, steps=1000,
                                            early_stopping_rounds=200)
classifier2.fit(X_train, y_train)
score2 = metrics.accuracy_score(y_test, classifier2.predict(X_test))

# you can expect the score is improved by using early stopping
print(score2 > score1)
