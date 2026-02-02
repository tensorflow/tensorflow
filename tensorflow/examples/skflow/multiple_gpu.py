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

from sklearn import datasets, metrics, cross_validation
import tensorflow as tf
from tensorflow.contrib import learn

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target,
    test_size=0.2, random_state=42)

def my_model(X, y):
    """
    This is DNN with 10, 20, 10 hidden layers, and dropout of 0.5 probability.

    Note: If you want to run this example with multiple GPUs, Cuda Toolkit 7.0 and 
    CUDNN 6.5 V2 from NVIDIA need to be installed beforehand. 
    """
    with tf.device('/gpu:1'):
    	layers = learn.ops.dnn(X, [10, 20, 10], dropout=0.5)
    with tf.device('/gpu:2'):
    	return learn.models.logistic_regression(layers, y)

classifier = learn.TensorFlowEstimator(model_fn=my_model, n_classes=3)
classifier.fit(X_train, y_train)
score = metrics.accuracy_score(y_test, classifier.predict(X_test))
print('Accuracy: {0:f}'.format(score))
