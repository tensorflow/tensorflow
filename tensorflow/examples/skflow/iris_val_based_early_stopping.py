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

"""Example of DNNClassifier for Iris plant dataset, with early stopping."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil

from sklearn import datasets
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import tensorflow as tf

from tensorflow.contrib import learn


def clean_folder(folder):
  """Cleans the given folder if it exists."""
  try:
    shutil.rmtree(folder)
  except OSError:
    pass


def main(unused_argv):
  iris = datasets.load_iris()
  x_train, x_test, y_train, y_test = train_test_split(
      iris.data, iris.target, test_size=0.2, random_state=42)

  x_train, x_val, y_train, y_val = train_test_split(
      x_train, y_train, test_size=0.2, random_state=42)
  val_monitor = learn.monitors.ValidationMonitor(
      x_val, y_val, early_stopping_rounds=200)

  model_dir = '/tmp/iris_model'
  clean_folder(model_dir)

  # classifier with early stopping on training data
  classifier1 = learn.DNNClassifier(
      feature_columns=learn.infer_real_valued_columns_from_input(x_train),
      hidden_units=[10, 20, 10], n_classes=3, model_dir=model_dir)
  classifier1.fit(x=x_train, y=y_train, steps=2000)
  score1 = metrics.accuracy_score(y_test, classifier1.predict(x_test))

  model_dir = '/tmp/iris_model_val'
  clean_folder(model_dir)

  # classifier with early stopping on validation data, save frequently for
  # monitor to pick up new checkpoints.
  classifier2 = learn.DNNClassifier(
      feature_columns=learn.infer_real_valued_columns_from_input(x_train),
      hidden_units=[10, 20, 10], n_classes=3, model_dir=model_dir,
      config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1))
  classifier2.fit(x=x_train, y=y_train, steps=2000, monitors=[val_monitor])
  score2 = metrics.accuracy_score(y_test, classifier2.predict(x_test))

  # In many applications, the score is improved by using early stopping
  print('score1: ', score1)
  print('score2: ', score2)
  print('score2 > score1: ', score2 > score1)


if __name__ == '__main__':
  tf.app.run()
