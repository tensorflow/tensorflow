# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for detecting free vars in tf.function."""

import unittest

from absl.testing import parameterized

from tensorflow.python.compat import v2_compat
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.platform import test


class ByRefCaptureTest(test.TestCase, parameterized.TestCase):

  @combinations.generate(
      combinations.combine(val_type=[int, constant_op.constant]))
  def test_direct_capture_mutation(self, val_type):
    x = val_type(1)

    @def_function.function
    def f():
      graph = ops.get_default_graph()
      cap_x = graph._experimental_capture_side_input_by_ref("x", lambda: x)
      return cap_x + 1

    self.assertEqual(f(), 2)
    x = val_type(2)
    self.assertEqual(f(), 3)

  @unittest.skip("By ref capture API does not work for nested tf.function.")
  def test_capture_in_nested_function(self):
    x = constant_op.constant(1)

    @def_function.function
    def f():
      graph = ops.get_default_graph()
      # Capture the same x for the outer tf.function
      graph._experimental_capture_side_input_by_ref("x", lambda: x)

      @def_function.function
      def g():
        graph = ops.get_default_graph()
        cap_x = graph._experimental_capture_side_input_by_ref("xx", lambda: x)
        return cap_x + 100

      return g()

    self.assertEqual(f(), 2)
    x = constant_op.constant(2)
    self.assertEqual(f(), 102)

  def test_capture_in_outer_function(self):
    x = 1

    def g():
      graph = ops.get_default_graph()
      cap_x = graph._experimental_capture_side_input_by_ref("x", lambda: x)
      return cap_x + 1

    @def_function.function
    def f():
      return g()

    self.assertEqual(f(), 2)
    x = 2
    self.assertEqual(f(), 3)

  @unittest.skip("By ref capture API does not work for nested tf.function.")
  def test_capture_in_outer_tf_function(self):
    x = 1

    @def_function.function
    def g():
      graph = ops.get_default_graph()
      cap_x = graph._experimental_capture_side_input_by_ref("x", lambda: x)
      return cap_x + 1

    @def_function.function
    def f():
      # Call `_experimental_capture_side_input_by_ref` so that the outer
      # tf.function will retrace when needed.
      graph = ops.get_default_graph()
      graph._experimental_capture_side_input_by_ref("x", lambda: x)
      return g()

    self.assertEqual(f(), 2)
    x = 2
    self.assertEqual(f(), 3)


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  test.main()
