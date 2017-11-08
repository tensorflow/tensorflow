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
"""Tests for third_party.tensorflow.contrib.quantize.python.quant_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.quantize.python import quant_ops
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest

_MIN_MAX_VARS = 'min_max_vars'


class QuantOpsTest(googletest.TestCase):

  def testLastValueQuantizeTrainingAssign(self):
    g = ops.Graph()
    with session.Session(graph=g) as sess:
      x = array_ops.placeholder(dtypes.float32, shape=[2])
      y = quant_ops.LastValueQuantize(
          x,
          init_min=0.0,
          init_max=0.0,
          is_training=True,
          vars_collection=_MIN_MAX_VARS)

      # Run the step.
      sess.run(variables.global_variables_initializer())
      sess.run(y, feed_dict={x: [-1.0, 1.0]})
      # Now check that the min_max_vars were, in fact, updated.
      min_value, max_value = self._GetMinMaxValues(sess)
      self.assertEqual(min_value, -1.0)
      self.assertEqual(max_value, 1.0)

  def testMovingAvgQuantizeTrainingAssign(self):
    g = ops.Graph()
    with session.Session(graph=g) as sess:
      x = array_ops.placeholder(dtypes.float32, shape=[2])
      y = quant_ops.MovingAvgQuantize(
          x,
          init_min=0.0,
          init_max=0.0,
          is_training=True,
          vars_collection=_MIN_MAX_VARS)

      # Run the step.
      sess.run(variables.global_variables_initializer())
      # Do two runs to avoid zero debias.
      sess.run(y, feed_dict={x: [-1.0, 1.0]})
      sess.run(y, feed_dict={x: [0.0, 0.0]})
      # Now check that the min_max_vars were, in fact, updated.
      min_value, max_value = self._GetMinMaxValues(sess)
      self.assertGreater(min_value, -1.0)
      self.assertLess(min_value, 0.0)
      self.assertGreater(max_value, 0.0)
      self.assertLess(max_value, 1.0)

  def _GetMinMaxValues(self, sess):
    min_max_vars = ops.get_collection(_MIN_MAX_VARS)
    self.assertEqual(len(min_max_vars), 2)
    min_idx = 0 if 'min' in min_max_vars[0].name else 1
    max_idx = (min_idx + 1) % 2
    min_var, max_var = min_max_vars[min_idx], min_max_vars[max_idx]
    min_max_values = sess.run([min_var, max_var])
    return min_max_values[0], min_max_values[1]


if __name__ == '__main__':
  googletest.main()
