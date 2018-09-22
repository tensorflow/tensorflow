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
"""Tests for state updating ops that may have benign race conditions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class AssignOpTest(test.TestCase):

  # NOTE(mrry): We exclude thess tests from the TSAN TAP target, because they
  #   contain benign and deliberate data races when multiple threads update
  #   the same parameters without a lock.
  def testParallelUpdateWithoutLocking(self):
    with self.cached_session() as sess:
      ones_t = array_ops.fill([1024, 1024], 1.0)
      p = variables.Variable(array_ops.zeros([1024, 1024]))
      adds = [
          state_ops.assign_add(
              p, ones_t, use_locking=False) for _ in range(20)
      ]
      variables.global_variables_initializer().run()

      def run_add(add_op):
        sess.run(add_op)

      threads = [
          self.checkedThread(
              target=run_add, args=(add_op,)) for add_op in adds
      ]
      for t in threads:
        t.start()
      for t in threads:
        t.join()

      vals = p.eval()
      ones = np.ones((1024, 1024)).astype(np.float32)
      self.assertTrue((vals >= ones).all())
      self.assertTrue((vals <= ones * 20).all())

  def testParallelAssignWithoutLocking(self):
    with self.cached_session() as sess:
      ones_t = array_ops.fill([1024, 1024], float(1))
      p = variables.Variable(array_ops.zeros([1024, 1024]))
      assigns = [
          state_ops.assign(p, math_ops.multiply(ones_t, float(i)), False)
          for i in range(1, 21)
      ]
      variables.global_variables_initializer().run()

      def run_assign(assign_op):
        sess.run(assign_op)

      threads = [
          self.checkedThread(
              target=run_assign, args=(assign_op,)) for assign_op in assigns
      ]
      for t in threads:
        t.start()
      for t in threads:
        t.join()

      vals = p.eval()

      # Assert every element is taken from one of the assignments.
      self.assertTrue((vals > 0).all())
      self.assertTrue((vals <= 20).all())

  # NOTE(skyewm): We exclude these tests from the TSAN TAP target, because they
  # contain non-benign but known data races between the variable assignment and
  # returning the output tensors. This issue will be resolved with the new
  # resource variables.
  def testParallelUpdateWithLocking(self):
    with self.cached_session() as sess:
      zeros_t = array_ops.fill([1024, 1024], 0.0)
      ones_t = array_ops.fill([1024, 1024], 1.0)
      p = variables.Variable(zeros_t)
      adds = [
          state_ops.assign_add(
              p, ones_t, use_locking=True) for _ in range(20)
      ]
      p.initializer.run()

      def run_add(add_op):
        sess.run(add_op)

      threads = [
          self.checkedThread(
              target=run_add, args=(add_op,)) for add_op in adds
      ]
      for t in threads:
        t.start()
      for t in threads:
        t.join()

      vals = p.eval()
      ones = np.ones((1024, 1024)).astype(np.float32)
      self.assertAllEqual(vals, ones * 20)

  def testParallelAssignWithLocking(self):
    with self.cached_session() as sess:
      zeros_t = array_ops.fill([1024, 1024], 0.0)
      ones_t = array_ops.fill([1024, 1024], 1.0)
      p = variables.Variable(zeros_t)
      assigns = [
          state_ops.assign(
              p, math_ops.multiply(ones_t, float(i)), use_locking=True)
          for i in range(1, 21)
      ]
      p.initializer.run()

      def run_assign(assign_op):
        sess.run(assign_op)

      threads = [
          self.checkedThread(
              target=run_assign, args=(assign_op,)) for assign_op in assigns
      ]
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
  test.main()
