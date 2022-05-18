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
"""Tests for side inputs in tf.function."""

import unittest
from absl.testing import parameterized

import tensorflow as tf


# TODO(panzf): Enable tests after side inputs are supported
class SideInputsTest(parameterized.TestCase):

  @parameterized.parameters(
      (1, tf.constant, 2, tf.constant),
      (1.0, tf.constant, 2.0, tf.constant),
      (1, int, 2, int),
      (1.0, float, 2.0, float),
      (1, int, 2, tf.constant),
      (1, tf.constant, 2, int))
  @unittest.skip("Feature not implemented")
  def test_direct_capture(self, val_before, type_before, val_after, type_after):
    def f():
      return x + tf.constant(1)

    tf_f = tf.function(f)
    x = type_before(val_before)
    self.assertEqual(f(), tf_f())
    x = type_after(val_after)
    self.assertEqual(f(), tf_f())

  @unittest.skip("Feature not implemented")
  def test_direct_capture_mutation(self):
    def f():
      return glob[-1] + tf.constant(0)

    tf_f = tf.function(f)
    glob = [tf.constant(1), tf.constant(2)]

    self.assertEqual(f(), tf_f())
    glob.append(tf.constant(3))
    self.assertEqual(f(), tf_f())

  @unittest.skip("Feature not implemented")
  @parameterized.parameters(
      tf.constant,
      int)
  def test_dict_capture_mutation_with_tensor_and_non_tensor(self, capture_type):
    def f():
      return d["val"]

    tf_f = tf.function(f)
    d = {"int": 1, "tensor": tf.constant(2), "val": capture_type(3)}
    self.assertEqual(f(), tf_f())
    d["val"] = capture_type(4)
    self.assertEqual(f(), tf_f())

  @unittest.skip("Feature not implemented")
  @parameterized.parameters(tf.constant, int)
  def test_capture_with_duplicate_usage(self, capture_type):
    def f():
      return x + x  # should capture x just once.

    tf_f = tf.function(f)

    x = capture_type(1)
    self.assertEqual(f(), tf_f())
    self.assertLen(tf_f.get_concrete_function().graph.inputs, 1)

    x = capture_type(2)
    self.assertEqual(f(), tf_f())
    self.assertLen(tf_f.get_concrete_function().graph.inputs, 1)

  @unittest.skip("Feature not implemented")
  def test_local_capture(self):
    @tf.function
    def f():
      x = tf.constant(0)
      def g():
        return x
      return g()

    tf_f = tf.function(f)
    x = tf.constant(100)  # pylint: disable=unused-variable
    self.assertEqual(f(), tf_f())
    x = tf.constant(200)
    self.assertEqual(f(), tf_f())

  @parameterized.parameters(
      tf.constant,
      int)
  @unittest.skip("Feature not implemented")
  def test_capture_by_nested_function(self, capture_type):
    @tf.function
    def f():
      def g():
        return x
      return g()

    tf_f = tf.function(f)

    x = capture_type(1)
    self.assertEqual(f(), tf_f())
    x = capture_type(2)
    self.assertEqual(f(), tf_f())

  @parameterized.parameters(tf.constant, int)
  @unittest.skip("Feature not implemented")
  def test_outer_capture_with_function_call(self, capture_type):
    def g():
      return x

    @tf.function
    def f():
      return g()
    tf_f = tf.function(f)

    x = capture_type(1)
    self.assertEqual(f(), tf_f())
    x = capture_type(2)
    self.assertEqual(f(), tf_f())

  @parameterized.parameters(tf.constant, int)
  @unittest.skip("Feature not implemented")
  def test_outer_capture_with_nested_function_call(self, capture_type):
    def g_factory():
      def g():
        return x
      return g()

    @tf.function
    def f():
      h = g_factory()
      return h()
    tf_f = tf.function(f)

    x = capture_type(1)
    self.assertEqual(f(), tf_f())
    x = capture_type(2)
    self.assertEqual(f(), tf_f())

  @parameterized.parameters(tf.constant, int)
  @unittest.skip("Feature not implemented")
  def test_capture_within_function_argument(self, capture_type):
    def g():
      return x

    @tf.function
    def f(h):
      return h()
    tf_f = tf.function(f)

    x = capture_type(1)
    self.assertEqual(f(g), tf_f(g))
    x = capture_type(2)
    self.assertEqual(f(g), tf_f(g))

  @parameterized.parameters(
      tf.constant,
      int)
  @unittest.skip("Feature not implemented")
  def test_nested_tf_function_with_capture(self, capture_type):
    @tf.function
    def tf_f():
      @tf.function
      def tf_g():
        return x
      return tf_g()

    x = capture_type(0)
    self.assertEqual(tf_f(), tf.constant(0))
    x = capture_type(1)
    self.assertEqual(tf_f(), tf.constant(0))
    # Test the outer function doesn't have any captures
    self.assertLen(tf_f.get_concrete_function().graph.capture, 1)


if __name__ == "__main__":
  unittest.main()
