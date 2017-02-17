# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

import sys

# TODO: #6568 Remove this hack that makes dlopen() not crash.
if hasattr(sys, 'getdlopenflags') and hasattr(sys, 'setdlopenflags'):
  import ctypes
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

from tensorflow.contrib.layers.python.layers import summaries as summaries_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class SummariesTest(test.TestCase):

  def test_summarize_scalar_tensor(self):
    with self.test_session():
      scalar_var = variables.Variable(1)
      summary_op = summaries_lib.summarize_tensor(scalar_var)
      self.assertEquals(summary_op.op.type, 'ScalarSummary')

  def test_summarize_multidim_tensor(self):
    with self.test_session():
      tensor_var = variables.Variable([1, 2, 3])
      summary_op = summaries_lib.summarize_tensor(tensor_var)
      self.assertEquals(summary_op.op.type, 'HistogramSummary')

  def test_summarize_activation(self):
    with self.test_session():
      var = variables.Variable(1)
      op = array_ops.identity(var, name='SummaryTest')
      summary_op = summaries_lib.summarize_activation(op)

      self.assertEquals(summary_op.op.type, 'HistogramSummary')
      names = [op.op.name for op in ops.get_collection(ops.GraphKeys.SUMMARIES)]
      self.assertEquals(len(names), 1)
      self.assertIn(u'SummaryTest/activation', names)

  def test_summarize_activation_relu(self):
    with self.test_session():
      var = variables.Variable(1)
      op = nn_ops.relu(var, name='SummaryTest')
      summary_op = summaries_lib.summarize_activation(op)

      self.assertEquals(summary_op.op.type, 'HistogramSummary')
      names = [op.op.name for op in ops.get_collection(ops.GraphKeys.SUMMARIES)]
      self.assertEquals(len(names), 2)
      self.assertIn(u'SummaryTest/zeros', names)
      self.assertIn(u'SummaryTest/activation', names)

  def test_summarize_activation_relu6(self):
    with self.test_session():
      var = variables.Variable(1)
      op = nn_ops.relu6(var, name='SummaryTest')
      summary_op = summaries_lib.summarize_activation(op)

      self.assertEquals(summary_op.op.type, 'HistogramSummary')
      names = [op.op.name for op in ops.get_collection(ops.GraphKeys.SUMMARIES)]
      self.assertEquals(len(names), 3)
      self.assertIn(u'SummaryTest/zeros', names)
      self.assertIn(u'SummaryTest/sixes', names)
      self.assertIn(u'SummaryTest/activation', names)

  def test_summarize_collection_regex(self):
    with self.test_session():
      var = variables.Variable(1)
      array_ops.identity(var, name='Test1')
      ops.add_to_collection('foo', array_ops.identity(var, name='Test2'))
      ops.add_to_collection('foo', array_ops.identity(var, name='Foobar'))
      ops.add_to_collection('foo', array_ops.identity(var, name='Test3'))
      summaries = summaries_lib.summarize_collection('foo', r'Test[123]')
      names = [op.op.name for op in summaries]
      self.assertEquals(len(names), 2)
      self.assertIn(u'Test2_summary', names)
      self.assertIn(u'Test3_summary', names)


if __name__ == '__main__':
  test.main()
