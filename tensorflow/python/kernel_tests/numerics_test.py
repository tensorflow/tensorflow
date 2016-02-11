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

"""Tests for tensorflow.ops.numerics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import control_flow_ops


class VerifyTensorAllFiniteTest(tf.test.TestCase):

  def testVerifyTensorAllFiniteSucceeds(self):
    x_shape = [5, 4]
    x = np.random.random_sample(x_shape).astype(np.float32)
    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        t = tf.constant(x, shape=x_shape, dtype=tf.float32)
        t_verified = tf.verify_tensor_all_finite(t, "Input is not a number.")
        self.assertAllClose(x, t_verified.eval())

  def testVerifyTensorAllFiniteFails(self):
    x_shape = [5, 4]
    x = np.random.random_sample(x_shape).astype(np.float32)
    my_msg = "Input is not a number."

    # Test NaN.
    x[0] = np.nan
    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        with self.assertRaisesOpError(my_msg):
          t = tf.constant(x, shape=x_shape, dtype=tf.float32)
          t_verified = tf.verify_tensor_all_finite(t, my_msg)
          t_verified.eval()

    # Test Inf.
    x[0] = np.inf
    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        with self.assertRaisesOpError(my_msg):
          t = tf.constant(x, shape=x_shape, dtype=tf.float32)
          t_verified = tf.verify_tensor_all_finite(t, my_msg)
          t_verified.eval()


class NumericsTest(tf.test.TestCase):

  def testInf(self):
    for use_gpu in [True, False]:
      with self.test_session(use_gpu=use_gpu, graph=tf.Graph()):
        t1 = tf.constant(1.0)
        t2 = tf.constant(0.0)
        a = tf.div(t1, t2)
        check = tf.add_check_numerics_ops()
        a = control_flow_ops.with_dependencies([check], a)
        with self.assertRaisesOpError("Inf"):
          a.eval()

  def testNaN(self):
    for use_gpu in [True, False]:
      with self.test_session(use_gpu=use_gpu, graph=tf.Graph()):
        t1 = tf.constant(0.0)
        t2 = tf.constant(0.0)
        a = tf.div(t1, t2)
        check = tf.add_check_numerics_ops()
        a = control_flow_ops.with_dependencies([check], a)
        with self.assertRaisesOpError("NaN"):
          a.eval()

  def testBoth(self):
    for use_gpu in [True, False]:
      with self.test_session(use_gpu=use_gpu, graph=tf.Graph()):
        t1 = tf.constant([1.0, 0.0])
        t2 = tf.constant([0.0, 0.0])
        a = tf.div(t1, t2)
        check = tf.add_check_numerics_ops()
        a = control_flow_ops.with_dependencies([check], a)
        with self.assertRaisesOpError("Inf and NaN"):
          a.eval()

  def testPassThrough(self):
    for use_gpu in [True, False]:
      with self.test_session(use_gpu=use_gpu, graph=tf.Graph()):
        t1 = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
        checked = tf.check_numerics(t1, message="pass through test")
        value = checked.eval()
        self.assertAllEqual(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), value)
        self.assertEqual([2, 3], checked.get_shape())


if __name__ == "__main__":
  tf.test.main()
