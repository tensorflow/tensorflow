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

"""Tests for tensorflow.ops.tf.Assign*."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class AssignOpTest(tf.test.TestCase):

  def _initAssignFetch(self, x, y, use_gpu=False):
    """Initialize a param to init and update it with y."""
    super(AssignOpTest, self).setUp()
    with self.test_session(use_gpu=use_gpu):
      p = tf.Variable(x)
      assign = tf.assign(p, y)
      p.initializer.run()
      new_value = assign.eval()
      return p.eval(), new_value

  def _initAssignAddFetch(self, x, y, use_gpu=False):
    """Initialize a param to init, and compute param += y."""
    with self.test_session(use_gpu=use_gpu):
      p = tf.Variable(x)
      add = tf.assign_add(p, y)
      p.initializer.run()
      new_value = add.eval()
      return p.eval(), new_value

  def _initAssignSubFetch(self, x, y, use_gpu=False):
    """Initialize a param to init, and compute param -= y."""
    with self.test_session(use_gpu=use_gpu):
      p = tf.Variable(x)
      sub = tf.assign_sub(p, y)
      p.initializer.run()
      new_value = sub.eval()
      return p.eval(), new_value

  def _testTypes(self, vals):
    for dtype in [np.float32, np.float64, np.int32, np.int64]:
      x = np.zeros(vals.shape).astype(dtype)
      y = vals.astype(dtype)
      var_value, op_value = self._initAssignFetch(x, y, use_gpu=False)
      self.assertAllEqual(y, var_value)
      self.assertAllEqual(y, op_value)
      var_value, op_value = self._initAssignAddFetch(x, y, use_gpu=False)
      self.assertAllEqual(x + y, var_value)
      self.assertAllEqual(x + y, op_value)
      var_value, op_value = self._initAssignSubFetch(x, y, use_gpu=False)
      self.assertAllEqual(x - y, var_value)
      self.assertAllEqual(x - y, op_value)
      if tf.test.is_built_with_cuda() and dtype in [np.float32, np.float64]:
        var_value, op_value = self._initAssignFetch(x, y, use_gpu=True)
        self.assertAllEqual(y, var_value)
        self.assertAllEqual(y, op_value)
        var_value, op_value = self._initAssignAddFetch(x, y, use_gpu=True)
        self.assertAllEqual(x + y, var_value)
        self.assertAllEqual(x + y, op_value)
        var_value, op_value = self._initAssignSubFetch(x, y, use_gpu=False)
        self.assertAllEqual(x - y, var_value)
        self.assertAllEqual(x - y, op_value)

  def testBasic(self):
    self._testTypes(np.arange(0, 20).reshape([4, 5]))

  def testAssignNonStrictShapeChecking(self):
    with self.test_session():
      data = tf.fill([1024, 1024], 0)
      p = tf.Variable([1])
      a = tf.assign(p, data, validate_shape=False)
      a.op.run()
      self.assertAllEqual(p.eval(), data.eval())

      # Assign to yet another shape
      data2 = tf.fill([10, 10], 1)
      a2 = tf.assign(p, data2, validate_shape=False)
      a2.op.run()
      self.assertAllEqual(p.eval(), data2.eval())

  def testInitRequiredAssignAdd(self):
    with self.test_session():
      p = tf.Variable(tf.fill([1024, 1024], 1),
                             tf.int32)
      a = tf.assign_add(p, tf.fill([1024, 1024], 0))
      with self.assertRaisesOpError("use uninitialized"):
        a.op.run()

  def testInitRequiredAssignSub(self):
    with self.test_session():
      p = tf.Variable(tf.fill([1024, 1024], 1),
                             tf.int32)
      a = tf.assign_sub(p, tf.fill([1024, 1024], 0))
      with self.assertRaisesOpError("use uninitialized"):
        a.op.run()

  # NOTE(mrry): See also
  #   dense_update_ops_no_tsan_test.AssignOpTest, which contains a benign
  #   data race and must run without TSAN.
  def testParallelUpdateWithLocking(self):
    with self.test_session() as sess:
      zeros_t = tf.fill([1024, 1024], 0.0)
      ones_t = tf.fill([1024, 1024], 1.0)
      p = tf.Variable(zeros_t)
      adds = [tf.assign_add(p, ones_t, use_locking=True)
              for _ in range(20)]
      p.initializer.run()

      def run_add(add_op):
        sess.run(add_op)
      threads = [
          self.checkedThread(target=run_add, args=(add_op,)) for add_op in adds]
      for t in threads:
        t.start()
      for t in threads:
        t.join()

      vals = p.eval()
      ones = np.ones((1024, 1024)).astype(np.float32)
      self.assertAllEqual(vals, ones * 20)

  # NOTE(mrry): See also
  #   dense_update_ops_no_tsan_test.[...].testParallelAssignWithoutLocking,
  #   which contains a benign data race and must run without TSAN.
  def testParallelAssignWithLocking(self):
    with self.test_session() as sess:
      zeros_t = tf.fill([1024, 1024], 0.0)
      ones_t = tf.fill([1024, 1024], 1.0)
      p = tf.Variable(zeros_t)
      assigns = [tf.assign(p, tf.mul(ones_t, float(i)),
                                  use_locking=True)
                 for i in range(1, 21)]
      p.initializer.run()

      def run_assign(assign_op):
        sess.run(assign_op)
      threads = [self.checkedThread(target=run_assign, args=(assign_op,))
                 for assign_op in assigns]
      for t in threads:
        t.start()
      for t in threads:
        t.join()

      vals = p.eval()

      # Assert every element is the same, and taken from one of the assignments.
      self.assertTrue(vals[0, 0] > 0)
      self.assertTrue(vals[0, 0] <= 20)
      self.assertAllEqual(vals, np.ones([1024, 1024]) * vals[0, 0])


if __name__ == "__main__":
  tf.test.main()
