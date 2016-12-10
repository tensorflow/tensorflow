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

"""Tests for export tools."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import tempfile
import numpy as np
import six

import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.utils import export
from tensorflow.contrib.session_bundle import manifest_pb2


_X_KEY = 'my_x_key'


_X_COLUMN = tf.contrib.layers.real_valued_column(_X_KEY, dimension=1)


def _training_input_fn():
  x = tf.random_uniform(shape=(1,), minval=0.0, maxval=1000.0)
  y = 2 * x + 3
  return {_X_KEY: x}, y


class ExportTest(tf.test.TestCase):

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

  def _assert_export(self, export_monitor, export_dir, expected_signature):
    self.assertTrue(tf.gfile.Exists(export_dir))
    # Only the written checkpoints are exported.
    self.assertTrue(tf.gfile.Exists(export_dir + '00000001/export'))
    self.assertTrue(tf.gfile.Exists(export_dir + '00000010/export'))
    self.assertEquals(six.b(os.path.join(export_dir, '00000010')),
                      export_monitor.last_export_dir)
    # Validate the signature
    signature = self._get_default_signature(export_dir + '00000010/export.meta')
    self.assertTrue(signature.HasField(expected_signature))

  def testExportMonitor_EstimatorProvidesSignature(self):
    random.seed(42)
    x = np.random.rand(1000)
    y = 2 * x + 3
    cont_features = [tf.contrib.layers.real_valued_column('', dimension=1)]
    regressor = learn.LinearRegressor(feature_columns=cont_features)
    export_dir = tempfile.mkdtemp() + 'export/'
    export_monitor = learn.monitors.ExportMonitor(
        every_n_steps=1, export_dir=export_dir, exports_to_keep=2)
    regressor.fit(x, y, steps=10, monitors=[export_monitor])
    self._assert_export(export_monitor, export_dir, 'regression_signature')

  def testExportMonitor(self):
    random.seed(42)
    x = np.random.rand(1000)
    y = 2 * x + 3
    cont_features = [tf.contrib.layers.real_valued_column('', dimension=1)]
    export_dir = tempfile.mkdtemp() + 'export/'
    export_monitor = learn.monitors.ExportMonitor(
        every_n_steps=1, export_dir=export_dir, exports_to_keep=2,
        signature_fn=export.generic_signature_fn)
    regressor = learn.LinearRegressor(feature_columns=cont_features)
    regressor.fit(x, y, steps=10, monitors=[export_monitor])
    self._assert_export(export_monitor, export_dir, 'generic_signature')

  def testExportMonitorInputFeatureKeyMissing(self):
    random.seed(42)
    def _export_input_fn():
      return {
          _X_KEY: tf.random_uniform(shape=(1,), minval=0.0, maxval=1000.0)
      }, None
    input_feature_key = 'my_example_key'
    monitor = learn.monitors.ExportMonitor(
        every_n_steps=1, export_dir=tempfile.mkdtemp() + 'export/',
        input_fn=_export_input_fn, input_feature_key=input_feature_key,
        exports_to_keep=2, signature_fn=export.generic_signature_fn)
    regressor = learn.LinearRegressor(feature_columns=[_X_COLUMN])
    with self.assertRaisesRegexp(KeyError, input_feature_key):
      regressor.fit(input_fn=_training_input_fn, steps=10, monitors=[monitor])

  def testExportMonitorInputFeatureKeyNoneNoFeatures(self):
    random.seed(42)
    input_feature_key = 'my_example_key'
    def _export_input_fn():
      return {input_feature_key: None}, None
    monitor = learn.monitors.ExportMonitor(
        every_n_steps=1, export_dir=tempfile.mkdtemp() + 'export/',
        input_fn=_export_input_fn, input_feature_key=input_feature_key,
        exports_to_keep=2, signature_fn=export.generic_signature_fn)
    regressor = learn.LinearRegressor(feature_columns=[_X_COLUMN])
    with self.assertRaisesRegexp(
        ValueError, 'features or examples must be defined'):
      regressor.fit(input_fn=_training_input_fn, steps=10, monitors=[monitor])

  def testExportMonitorInputFeatureKeyNone(self):
    random.seed(42)
    input_feature_key = 'my_example_key'
    def _export_input_fn():
      return {
          input_feature_key: None,
          _X_KEY: tf.random_uniform(shape=(1,), minval=0.0, maxval=1000.0)
      }, None
    monitor = learn.monitors.ExportMonitor(
        every_n_steps=1, export_dir=tempfile.mkdtemp() + 'export/',
        input_fn=_export_input_fn, input_feature_key=input_feature_key,
        exports_to_keep=2, signature_fn=export.generic_signature_fn)
    regressor = learn.LinearRegressor(feature_columns=[_X_COLUMN])
    with self.assertRaisesRegexp(ValueError, 'examples cannot be None'):
      regressor.fit(input_fn=_training_input_fn, steps=10, monitors=[monitor])

  def testExportMonitorInputFeatureKeyNoFeatures(self):
    random.seed(42)
    input_feature_key = 'my_example_key'
    def _export_input_fn():
      return {
          input_feature_key: tf.placeholder(dtype=tf.string, shape=(1,))
      }, None
    monitor = learn.monitors.ExportMonitor(
        every_n_steps=1, export_dir=tempfile.mkdtemp() + 'export/',
        input_fn=_export_input_fn, input_feature_key=input_feature_key,
        exports_to_keep=2, signature_fn=export.generic_signature_fn)
    regressor = learn.LinearRegressor(feature_columns=[_X_COLUMN])
    with self.assertRaisesRegexp(KeyError, _X_KEY):
      regressor.fit(input_fn=_training_input_fn, steps=10, monitors=[monitor])

  def testExportMonitorInputFeature(self):
    random.seed(42)
    input_feature_key = 'my_example_key'
    def _export_input_fn():
      return {
          input_feature_key: tf.placeholder(dtype=tf.string, shape=(1,)),
          _X_KEY: tf.random_uniform(shape=(1,), minval=0.0, maxval=1000.0)
      }, None
    export_dir = tempfile.mkdtemp() + 'export/'
    monitor = learn.monitors.ExportMonitor(
        every_n_steps=1, export_dir=export_dir,
        input_fn=_export_input_fn, input_feature_key=input_feature_key,
        exports_to_keep=2, signature_fn=export.generic_signature_fn)
    regressor = learn.LinearRegressor(feature_columns=[_X_COLUMN])
    regressor.fit(input_fn=_training_input_fn, steps=10, monitors=[monitor])
    self._assert_export(monitor, export_dir, 'generic_signature')

  def testExportMonitorRegressionSignature(self):

    def _regression_signature(examples, unused_features, predictions):
      signatures = {}
      signatures['regression'] = (
          tf.contrib.session_bundle.exporter.regression_signature(examples,
                                                                  predictions))
      return signatures['regression'], signatures

    random.seed(42)
    x = np.random.rand(1000)
    y = 2 * x + 3
    cont_features = [tf.contrib.layers.real_valued_column('', dimension=1)]
    regressor = learn.LinearRegressor(feature_columns=cont_features)
    export_dir = tempfile.mkdtemp() + 'export/'
    export_monitor = learn.monitors.ExportMonitor(
        every_n_steps=1,
        export_dir=export_dir,
        exports_to_keep=1,
        signature_fn=_regression_signature)
    regressor.fit(x, y, steps=10, monitors=[export_monitor])

    self.assertTrue(tf.gfile.Exists(export_dir))
    self.assertFalse(tf.gfile.Exists(export_dir + '00000000/export'))
    self.assertTrue(tf.gfile.Exists(export_dir + '00000010/export'))
    # Validate the signature
    signature = self._get_default_signature(export_dir + '00000010/export.meta')
    self.assertTrue(signature.HasField('regression_signature'))

if __name__ == '__main__':
  tf.test.main()
