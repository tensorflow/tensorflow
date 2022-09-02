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
import numpy as np

from tensorflow.core.function.capture import free_vars_detect


# TODO(panzf): Enable tests after the detection function is added.
class FreeVarDetectionTest(parameterized.TestCase):

  @unittest.skip("Feature not implemented")
  def test_func_arg(self):
    x = 1  # pylint: disable=unused-variable

    def f(x):
      return x + 1

    free_vars, _, _ = free_vars_detect.detect_function_free_vars(f)
    self.assertEmpty(free_vars)

  @unittest.skip("Feature not implemented")
  def test_func_local_var(self):

    def f():
      x = 1
      return x + 1

    free_vars, _, _ = free_vars_detect.detect_function_free_vars(f)
    self.assertEmpty(free_vars)

  @unittest.skip("Feature not implemented")
  def test_global_var_int(self):
    x = 1

    def f():
      return x + 1

    free_vars, _, _ = free_vars_detect.detect_function_free_vars(f)
    self.assertSequenceEqual(free_vars, ["x"])

  @unittest.skip("Feature not implemented")
  def test_global_var_dict(self):
    glob = {"a": 1}

    def f():
      return glob["a"] + 1

    free_vars, _, _ = free_vars_detect.detect_function_free_vars(f)
    self.assertSequenceEqual(free_vars, ["glob"])

  @unittest.skip("Feature not implemented")
  def test_duplicate_global_var(self):
    x = 1

    def f():
      return x + x

    free_vars, _, _ = free_vars_detect.detect_function_free_vars(f)
    self.assertSequenceEqual(free_vars, ["x"])

  @unittest.skip("Feature not implemented")
  def test_glob_numpy_var(self):
    a = 0
    b = np.asarray(1)

    def f():
      c = np.asarray(2)
      res = a + b + c
      return res

    free_vars, _, _ = free_vars_detect.detect_function_free_vars(f)
    self.assertSequenceEqual(free_vars, ["a", "b"])

  @unittest.skip("Feature not implemented")
  def test_global_var_in_nested_func(self):
    x = 1

    def f():

      def g():
        return x + 1

      return g()

    free_vars, _, _ = free_vars_detect.detect_function_free_vars(f)
    self.assertSequenceEqual(free_vars, ["x"])

  @unittest.skip("Feature not implemented")
  def test_global_var_from_outer_func(self):
    x = 1

    def g():
      return x + 1

    def f():
      return g()

    free_vars, _, _ = free_vars_detect.detect_function_free_vars(f)
    self.assertSequenceEqual(free_vars, ["x"])

  @unittest.skip("Feature not implemented")
  def test_global_var_from_renamed_outer_func(self):
    x = 1

    def g():
      return x + 1

    def f():
      h = g
      return h()

    free_vars, _, _ = free_vars_detect.detect_function_free_vars(f)
    self.assertSequenceEqual(free_vars, ["x"])

  # TODO(panzf): Enable this test after support inspecting callable func args
  @unittest.skip("Feature not implemented")
  def test_global_var_from_arg_func(self):
    # This test is kept to record this pattern.
    x = 1

    def g():
      return x + 1

    def f(h):
      return h()

    _ = f(g)

  # TODO(panzf): add tests for file_names and line_nums


if __name__ == "__main__":
  unittest.main()
