# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Early stopping tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import tensorflow as tf

from tensorflow.contrib.learn.python import learn
from tensorflow.contrib.learn.python.learn import datasets
from tensorflow.contrib.learn.python.learn.estimators._sklearn import accuracy_score
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split


def _get_summary_events(folder):
  if not tf.gfile.Exists(folder):
    raise ValueError('Folder %s doesn\'t exist.' % folder)
  return tf.contrib.testing.latest_summaries(folder)


class EarlyStoppingTest(tf.test.TestCase):
  """Early stopping tests."""

  def testIrisES(self):
    random.seed(42)

    iris = datasets.load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42)
    val_monitor = learn.monitors.ValidationMonitor(
        x_val, y_val, every_n_steps=50, early_stopping_rounds=100,
        early_stopping_metric='accuracy', early_stopping_metric_minimize=False)

    # classifier without early stopping - overfitting
    classifier1 = learn.TensorFlowDNNClassifier(
        hidden_units=[10, 20, 10], n_classes=3, steps=1000)
    classifier1.fit(x_train, y_train)
    _ = accuracy_score(y_test, classifier1.predict(x_test))

    # Full 1000 steps, 12 summaries and no evaluation summary.
    # 12 summaries = global_step + first + every 100 out of 1000 steps.
    self.assertEqual(12, len(_get_summary_events(classifier1.model_dir)))
    with self.assertRaises(ValueError):
      _get_summary_events(classifier1.model_dir + '/eval')

    # classifier with early stopping - improved accuracy on testing set
    classifier2 = learn.TensorFlowDNNClassifier(
        hidden_units=[10, 20, 10], n_classes=3, steps=2000,
        config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1))

    classifier2.fit(x_train, y_train, monitors=[val_monitor])
    _ = accuracy_score(y_val, classifier2.predict(x_val))
    _ = accuracy_score(y_test, classifier2.predict(x_test))

    # Note, this test is unstable, so not checking for equality.
    # See stability_test for examples of stability issues.
    if val_monitor.early_stopped:
      self.assertLess(val_monitor.best_step, 2000)
      # Note, due to validation monitor stopping after the best score occur,
      # the accuracy at current checkpoint is less.
      # TODO(ipolosukhin): Time machine for restoring old checkpoints?
      # flaky, still not always best_value better then score2 value.
      # self.assertGreater(val_monitor.best_value, score2_val)

      # Early stopped, unstable so checking only < then max.
      self.assertLess(len(_get_summary_events(classifier2.model_dir)), 21)
      # Eval typically has ~6 events, but it varies based on the run.
      self.assertLess(len(_get_summary_events(
          classifier2.model_dir + '/eval')), 8)

    # TODO(ipolosukhin): Restore this?
    # self.assertGreater(score2, score1, "No improvement using early stopping.")


if __name__ == '__main__':
  tf.test.main()
