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
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.utils import export
from tensorflow.contrib.session_bundle import manifest_pb2


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

  def testExportMonitor(self):
    random.seed(42)
    x = np.random.rand(1000)
    y = 2 * x + 3
    cont_features = [tf.contrib.layers.real_valued_column('', dimension=1)]
    regressor = learn.LinearRegressor(feature_columns=cont_features)
    export_dir = tempfile.mkdtemp() + 'export/'
    export_monitor = learn.monitors.ExportMonitor(
        every_n_steps=1, export_dir=export_dir, exports_to_keep=2,
        signature_fn=export.generic_signature_fn)
    regressor.fit(x, y, steps=10,
                  monitors=[export_monitor])

    self.assertTrue(tf.gfile.Exists(export_dir))
    # Only the written checkpoints are exported.
    self.assertTrue(tf.gfile.Exists(export_dir + '00000001/export'))
    self.assertTrue(tf.gfile.Exists(export_dir + '00000010/export'))
    self.assertEquals(export_monitor.last_export_dir, os.path.join(export_dir,
                                                                   '00000010'))
    # Validate the signature
    signature = self._get_default_signature(export_dir + '00000010/export.meta')
    self.assertTrue(signature.HasField('generic_signature'))

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
