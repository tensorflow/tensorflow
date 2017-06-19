# -*- coding: utf-8 -*-
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Integration tests for the Scalars Plugin."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os.path

from six import StringIO
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.tensorboard.backend.event_processing import event_multiplexer
from tensorflow.tensorboard.plugins.scalars import scalars_plugin


class ScalarsPluginTest(tf.test.TestCase):

  _STEPS = 99

  _SCALAR_TAG = 'simple-values'
  _HISTOGRAM_TAG = 'complicated-values'

  _RUN_WITH_SCALARS = '_RUN_WITH_SCALARS'
  _RUN_WITH_HISTOGRAM = '_RUN_WITH_HISTOGRAM'

  def set_up_with_runs(self, run_names):
    self.logdir = self.get_temp_dir()
    for run_name in run_names:
      self.generate_run(run_name)
    multiplexer = event_multiplexer.EventMultiplexer()
    multiplexer.AddRunsFromDirectory(self.logdir)
    multiplexer.Reload()
    self.plugin = scalars_plugin.ScalarsPlugin()
    self.apps = self.plugin.get_plugin_apps(multiplexer, None)

  def generate_run(self, run_name):
    if run_name == self._RUN_WITH_SCALARS:
      (use_scalars, use_histogram) = (True, False)
    elif run_name == self._RUN_WITH_HISTOGRAM:
      (use_scalars, use_histogram) = (False, True)
    else:
      assert False, 'Invalid run name: %r' % run_name
    tf.reset_default_graph()
    sess = tf.Session()
    if use_scalars:
      scalar_placeholder = tf.placeholder(tf.int64)
      tf.summary.scalar(self._SCALAR_TAG, scalar_placeholder)
    if use_histogram:
      histogram_placeholder = tf.placeholder(tf.float32, shape=[3])
      tf.summary.histogram(self._HISTOGRAM_TAG, histogram_placeholder)
    summ = tf.summary.merge_all()

    subdir = os.path.join(self.logdir, run_name)
    writer = tf.summary.FileWriter(subdir)
    writer.add_graph(sess.graph)
    for step in xrange(self._STEPS):
      feed_dict = {}
      if use_scalars:
        feed_dict[scalar_placeholder] = int((43**step) % 47)
      if use_histogram:
        feed_dict[histogram_placeholder] = [1 + step, 2 + step, 3 + step]
      s = sess.run(summ, feed_dict=feed_dict)
      writer.add_summary(s, global_step=step)
    writer.close()

  def test_index(self):
    self.set_up_with_runs([self._RUN_WITH_SCALARS, self._RUN_WITH_HISTOGRAM])
    self.assertEqual({
        self._RUN_WITH_SCALARS: [self._SCALAR_TAG],
        self._RUN_WITH_HISTOGRAM: [],
    }, self.plugin.index_impl())

  def _test_scalars_json(self, run_name, should_have_scalars):
    self.set_up_with_runs([self._RUN_WITH_SCALARS, self._RUN_WITH_HISTOGRAM])
    if should_have_scalars:
      (data, mime_type) = self.plugin.scalars_impl(
          self._SCALAR_TAG, run_name, scalars_plugin.OutputFormat.JSON)
      self.assertEqual('application/json', mime_type)
      self.assertEqual(len(data), self._STEPS)
    else:
      with self.assertRaises(KeyError):
        self.plugin.scalars_impl(self._SCALAR_TAG, run_name,
                                 scalars_plugin.OutputFormat.JSON)

  def _test_scalars_csv(self, run_name, should_have_scalars):
    self.set_up_with_runs([self._RUN_WITH_SCALARS, self._RUN_WITH_HISTOGRAM])
    if should_have_scalars:
      (data, mime_type) = self.plugin.scalars_impl(
          self._SCALAR_TAG, run_name, scalars_plugin.OutputFormat.CSV)
      self.assertEqual('text/csv', mime_type)
      s = StringIO(data)
      reader = csv.reader(s)
      self.assertEqual(['Wall time', 'Step', 'Value'], next(reader))
      self.assertEqual(len(list(reader)), self._STEPS)
    else:
      with self.assertRaises(KeyError):
        self.plugin.scalars_impl(self._SCALAR_TAG, run_name,
                                 scalars_plugin.OutputFormat.CSV)

  def test_scalars_json_with_scalars(self):
    self._test_scalars_json(self._RUN_WITH_SCALARS, True)

  def test_scalars_json_with_histogram(self):
    self._test_scalars_json(self._RUN_WITH_HISTOGRAM, False)

  def test_scalars_csv_with_scalars(self):
    self._test_scalars_csv(self._RUN_WITH_SCALARS, True)

  def test_scalars_csv_with_histogram(self):
    self._test_scalars_csv(self._RUN_WITH_HISTOGRAM, False)

  def test_active_with_scalars(self):
    self.set_up_with_runs([self._RUN_WITH_SCALARS])
    self.assertTrue(self.plugin.is_active())

  def test_active_with_histogram(self):
    self.set_up_with_runs([self._RUN_WITH_HISTOGRAM])
    self.assertFalse(self.plugin.is_active())

  def test_active_with_both(self):
    self.set_up_with_runs([self._RUN_WITH_SCALARS, self._RUN_WITH_HISTOGRAM])
    self.assertTrue(self.plugin.is_active())


if __name__ == '__main__':
  tf.test.main()
