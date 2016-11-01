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

"""Tests for Classifier."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tempfile

import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import _sklearn
from tensorflow.contrib.session_bundle import manifest_pb2


def iris_input_fn(num_epochs=None):
  iris = tf.contrib.learn.datasets.load_iris()
  features = tf.train.limit_epochs(
      tf.reshape(tf.constant(iris.data), [-1, 4]), num_epochs=num_epochs)
  labels = tf.reshape(tf.constant(iris.target), [-1])
  return features, labels


def logistic_model_fn(features, labels, unused_mode):
  labels = tf.one_hot(labels, 3, 1, 0)
  prediction, loss = tf.contrib.learn.models.logistic_regression_zero_init(
      features, labels)
  train_op = tf.contrib.layers.optimize_loss(
      loss, tf.contrib.framework.get_global_step(), optimizer='Adagrad',
      learning_rate=0.1)
  return prediction, loss, train_op


def logistic_model_params_fn(features, labels, unused_mode, params):
  labels = tf.one_hot(labels, 3, 1, 0)
  prediction, loss = tf.contrib.learn.models.logistic_regression_zero_init(
      features, labels)
  train_op = tf.contrib.layers.optimize_loss(
      loss, tf.contrib.framework.get_global_step(), optimizer='Adagrad',
      learning_rate=params['learning_rate'])
  return prediction, loss, train_op


class ClassifierTest(tf.test.TestCase):

  def testIrisAll(self):
    est = tf.contrib.learn.Classifier(model_fn=logistic_model_fn, n_classes=3)
    self._runIrisAll(est)

  def testIrisAllWithParams(self):
    est = tf.contrib.learn.Classifier(model_fn=logistic_model_params_fn,
                                      n_classes=3,
                                      params={'learning_rate': 0.01})
    self._runIrisAll(est)

  def testIrisInputFn(self):
    iris = tf.contrib.learn.datasets.load_iris()
    est = tf.contrib.learn.Classifier(model_fn=logistic_model_fn, n_classes=3)
    est.fit(input_fn=iris_input_fn, steps=100)
    est.evaluate(input_fn=iris_input_fn, steps=1, name='eval')
    predict_input_fn = functools.partial(iris_input_fn, num_epochs=1)
    predictions = list(est.predict(input_fn=predict_input_fn))
    self.assertEqual(len(predictions), iris.target.shape[0])

  def _runIrisAll(self, est):
    iris = tf.contrib.learn.datasets.load_iris()
    est.fit(iris.data, iris.target, steps=100)
    scores = est.evaluate(x=iris.data, y=iris.target, name='eval')
    predictions = list(est.predict(x=iris.data))
    predictions_proba = list(est.predict_proba(x=iris.data))
    self.assertEqual(len(predictions), iris.target.shape[0])
    self.assertAllEqual(predictions, np.argmax(predictions_proba, axis=1))
    other_score = _sklearn.accuracy_score(iris.target, predictions)
    self.assertAllClose(other_score, scores['accuracy'])

  def _get_default_signature(self, export_meta_filename):
    """Gets the default signature from the export.meta file."""
    with tf.Session():
      save = tf.train.import_meta_graph(export_meta_filename)
      meta_graph_def = save.export_meta_graph()
      collection_def = meta_graph_def.collection_def

      signatures_any = collection_def['serving_signatures'].any_list.value
      self.assertEquals(len(signatures_any), 1)
      signatures = manifest_pb2.Signatures()
      signatures_any[0].Unpack(signatures)
      default_signature = signatures.default_signature
      return default_signature

  # Disable this test case until b/31032996 is fixed.
  def _testExportMonitorRegressionSignature(self):
    iris = tf.contrib.learn.datasets.load_iris()
    est = tf.contrib.learn.Classifier(model_fn=logistic_model_fn, n_classes=3)
    export_dir = tempfile.mkdtemp() + 'export/'
    export_monitor = learn.monitors.ExportMonitor(
        every_n_steps=1,
        export_dir=export_dir,
        exports_to_keep=1,
        signature_fn=tf.contrib.learn.classifier.classification_signature_fn)
    est.fit(iris.data, iris.target, steps=2, monitors=[export_monitor])

    self.assertTrue(tf.gfile.Exists(export_dir))
    self.assertFalse(tf.gfile.Exists(export_dir + '00000000/export'))
    self.assertTrue(tf.gfile.Exists(export_dir + '00000002/export'))
    # Validate the signature
    signature = self._get_default_signature(export_dir + '00000002/export.meta')
    self.assertTrue(signature.HasField('classification_signature'))


if __name__ == '__main__':
  tf.test.main()
