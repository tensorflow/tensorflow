# Copyright 2015 Google Inc. All Rights Reserved.
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
"""Tests for regularizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class SummariesTest(tf.test.TestCase):

  def test_duplicate_tag(self):
    with self.test_session():
      var = tf.Variable([1, 2, 3])
      tf.contrib.layers.summarize_tensor(var)
      with self.assertRaises(ValueError):
        tf.contrib.layers.summarize_tensor(var)

  def test_summarize_scalar_tensor(self):
    with self.test_session():
      scalar_var = tf.Variable(1)
      summary_op = tf.contrib.layers.summarize_tensor(scalar_var)
      self.assertTrue(summary_op.op.type == 'ScalarSummary')

  def test_summarize_multidim_tensor(self):
    with self.test_session():
      tensor_var = tf.Variable([1, 2, 3])
      summary_op = tf.contrib.layers.summarize_tensor(tensor_var)
      self.assertTrue(summary_op.op.type == 'HistogramSummary')

  def test_summarize_activation(self):
    with self.test_session():
      var = tf.Variable(1)
      op = tf.identity(var, name='SummaryTest')
      summary_op = tf.contrib.layers.summarize_activation(op)

      self.assertTrue(summary_op.op.type == 'HistogramSummary')
      names = [op.op.name for op in tf.get_collection(tf.GraphKeys.SUMMARIES)]
      self.assertEquals(len(names), 1)
      self.assertTrue(u'SummaryTest/activation_summary' in names)

  def test_summarize_activation_relu(self):
    with self.test_session():
      var = tf.Variable(1)
      op = tf.nn.relu(var, name='SummaryTest')
      summary_op = tf.contrib.layers.summarize_activation(op)

      self.assertTrue(summary_op.op.type == 'HistogramSummary')
      names = [op.op.name for op in tf.get_collection(tf.GraphKeys.SUMMARIES)]
      self.assertEquals(len(names), 2)
      self.assertTrue(u'SummaryTest/zeros_summary' in names)
      self.assertTrue(u'SummaryTest/activation_summary' in names)

  def test_summarize_activation_relu6(self):
    with self.test_session():
      var = tf.Variable(1)
      op = tf.nn.relu6(var, name='SummaryTest')
      summary_op = tf.contrib.layers.summarize_activation(op)

      self.assertTrue(summary_op.op.type == 'HistogramSummary')
      names = [op.op.name for op in tf.get_collection(tf.GraphKeys.SUMMARIES)]
      self.assertEquals(len(names), 3)
      self.assertTrue(u'SummaryTest/zeros_summary' in names)
      self.assertTrue(u'SummaryTest/sixes_summary' in names)
      self.assertTrue(u'SummaryTest/activation_summary' in names)

  def test_summarize_collection_regex(self):
    with self.test_session():
      var = tf.Variable(1)
      tf.identity(var, name='Test1')
      tf.add_to_collection('foo', tf.identity(var, name='Test2'))
      tf.add_to_collection('foo', tf.identity(var, name='Foobar'))
      tf.add_to_collection('foo', tf.identity(var, name='Test3'))
      summaries = tf.contrib.layers.summarize_collection('foo', r'Test[123]')
      names = [op.op.name for op in summaries]
      self.assertEquals(len(names), 2)
      self.assertTrue(u'Test2_summary' in names)
      self.assertTrue(u'Test3_summary' in names)

if __name__ == '__main__':
  tf.test.main()
