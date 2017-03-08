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

"""Example of DNNClassifier for Iris plant dataset, h5 format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn import cross_validation
from sklearn import metrics
import tensorflow as tf
from tensorflow.contrib import learn
import h5py  # pylint: disable=g-bad-import-order


def main(unused_argv):
  # Load dataset.
  iris = learn.datasets.load_dataset('iris')
  x_train, x_test, y_train, y_test = cross_validation.train_test_split(
      iris.data, iris.target, test_size=0.2, random_state=42)

  # Note that we are saving and load iris data as h5 format as a simple
  # demonstration here.
  h5f = h5py.File('/tmp/test_hdf5.h5', 'w')
  h5f.create_dataset('X_train', data=x_train)
  h5f.create_dataset('X_test', data=x_test)
  h5f.create_dataset('y_train', data=y_train)
  h5f.create_dataset('y_test', data=y_test)
  h5f.close()

  h5f = h5py.File('/tmp/test_hdf5.h5', 'r')
  x_train = np.array(h5f['X_train'])
  x_test = np.array(h5f['X_test'])
  y_train = np.array(h5f['y_train'])
  y_test = np.array(h5f['y_test'])

  # Build 3 layer DNN with 10, 20, 10 units respectively.
  feature_columns = learn.infer_real_valued_columns_from_input(x_train)
  classifier = learn.DNNClassifier(
      feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=3)

  # Fit and predict.
  classifier.fit(x_train, y_train, steps=200)
  score = metrics.accuracy_score(y_test, classifier.predict(x_test))
  print('Accuracy: {0:f}'.format(score))

if __name__ == '__main__':
  tf.app.run()
