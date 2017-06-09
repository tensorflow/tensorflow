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
"""Integration tests for the Histograms Plugin."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.tensorboard.backend.event_processing import event_accumulator
from tensorflow.tensorboard.backend.event_processing import event_multiplexer
from tensorflow.tensorboard.plugins.histograms import histograms_plugin


class HistogramsPluginTest(tf.test.TestCase):

  _STEPS = 99

  _HISTOGRAM_TAG = 'my-favorite-histogram'
  _SCALAR_TAG = 'my-boring-scalars'

  _RUN_WITH_HISTOGRAM = '_RUN_WITH_HISTOGRAM'
  _RUN_WITH_SCALARS = '_RUN_WITH_SCALARS'

  def set_up_with_runs(self, run_names):
    self.logdir = self.get_temp_dir()
    for run_name in run_names:
      self.generate_run(run_name)
    multiplexer = event_multiplexer.EventMultiplexer(size_guidance={
        # don't truncate my test data, please
        event_accumulator.HISTOGRAMS:
            self._STEPS,
    })
    multiplexer.AddRunsFromDirectory(self.logdir)
    multiplexer.Reload()
    self.plugin = histograms_plugin.HistogramsPlugin()
    self.apps = self.plugin.get_plugin_apps(multiplexer, None)

  def generate_run(self, run_name):
    if run_name == self._RUN_WITH_HISTOGRAM:
      (use_histogram, use_scalars) = (True, False)
    elif run_name == self._RUN_WITH_SCALARS:
      (use_histogram, use_scalars) = (False, True)
    else:
      assert False, 'Invalid run name: %r' % run_name
    tf.reset_default_graph()
    sess = tf.Session()
    placeholder = tf.placeholder(tf.float32, shape=[3])
    if use_histogram:
      tf.summary.histogram(self._HISTOGRAM_TAG, placeholder)
    if use_scalars:
      tf.summary.scalar(self._SCALAR_TAG, tf.reduce_mean(placeholder))
    summ = tf.summary.merge_all()

    subdir = os.path.join(self.logdir, run_name)
    writer = tf.summary.FileWriter(subdir)
    writer.add_graph(sess.graph)
    for step in xrange(self._STEPS):
      feed_dict = {placeholder: [1 + step, 2 + step, 3 + step]}
      s = sess.run(summ, feed_dict=feed_dict)
      writer.add_summary(s, global_step=step)
    writer.close()

  def test_index(self):
    self.set_up_with_runs([self._RUN_WITH_HISTOGRAM, self._RUN_WITH_SCALARS])
    self.assertEqual({
        self._RUN_WITH_HISTOGRAM: [self._HISTOGRAM_TAG],
        self._RUN_WITH_SCALARS: [],
    }, self.plugin.index_impl())

  def _test_histograms(self, run_name, should_have_histogram):
    self.set_up_with_runs([self._RUN_WITH_HISTOGRAM, self._RUN_WITH_SCALARS])
    if should_have_histogram:
      (data, mime_type) = self.plugin.histograms_impl(self._HISTOGRAM_TAG,
                                                      run_name)
      self.assertEqual('application/json', mime_type)
      self.assertEqual(len(data), self._STEPS)
      for i in xrange(self._STEPS):
        frame = data[i]
        self.assertEqual(i, frame.step)
        self.assertEqual(1 + i, frame.histogram_value.min)
        self.assertEqual(3 + i, frame.histogram_value.max)
        self.assertAlmostEqual(
            3,  # three items across all buckets
            sum(frame.histogram_value.bucket))
    else:
      with self.assertRaises(KeyError):
        self.plugin.histograms_impl(self._HISTOGRAM_TAG, run_name)

  def test_histograms_with_scalars(self):
    self._test_histograms(self._RUN_WITH_HISTOGRAM, True)

  def test_histograms_with_histogram(self):
    self._test_histograms(self._RUN_WITH_SCALARS, False)

  def test_active_with_histogram(self):
    self.set_up_with_runs([self._RUN_WITH_HISTOGRAM])
    self.assertTrue(self.plugin.is_active())

  def test_active_with_scalars(self):
    self.set_up_with_runs([self._RUN_WITH_SCALARS])
    self.assertFalse(self.plugin.is_active())

  def test_active_with_both(self):
    self.set_up_with_runs([self._RUN_WITH_HISTOGRAM, self._RUN_WITH_SCALARS])
    self.assertTrue(self.plugin.is_active())


if __name__ == '__main__':
  tf.test.main()
