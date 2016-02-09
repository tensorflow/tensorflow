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

from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import skflow

iris = load_iris()
X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target,
    test_size=0.2, random_state=42)

# It's useful to scale to ensure Stochastic Gradient Descent will do the right thing
scaler = StandardScaler()

# DNN classifier
DNNclassifier = skflow.TensorFlowDNNClassifier(hidden_units=[10, 20, 10], n_classes=3, steps=200)

pipeline = Pipeline([('scaler', StandardScaler()), ('DNNclassifier', DNNclassifier)])

pipeline.fit(X_train, y_train)

score = accuracy_score(y_test, pipeline.predict(X_test))

print('Accuracy: {0:f}'.format(score))
