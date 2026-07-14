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
"""Tests for the experimental input pipeline ops."""
from absl.testing import parameterized

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class GetSingleElementTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              skip=[0, 5, 10], take=[1], error=[None], error_msg=[None]) +
          combinations.combine(
              skip=[100],
              take=[1],
              error=[errors.InvalidArgumentError],
              error_msg=["Dataset was empty."]) + combinations.combine(
                  skip=[0],
                  take=[2],
                  error=[errors.InvalidArgumentError],
                  error_msg=["Dataset had more than one element."])))
  def testBasic(self, skip, take, error=None, error_msg=None):

    def make_sparse(x):
      x_1d = array_ops.reshape(x, [1])
      x_2d = array_ops.reshape(x, [1, 1])
      return sparse_tensor.SparseTensor(x_2d, x_1d, x_1d)

    dataset = dataset_ops.Dataset.range(100).skip(skip).map(
        lambda x: (x * x, make_sparse(x))).take(take)
    if error is None:
      dense_val, sparse_val = self.evaluate(dataset.get_single_element())
      self.assertEqual(skip * skip, dense_val)
      self.assertAllEqual([[skip]], sparse_val.indices)
      self.assertAllEqual([skip], sparse_val.values)
      self.assertAllEqual([skip], sparse_val.dense_shape)
    else:
      with self.assertRaisesRegex(error, error_msg):
        self.evaluate(dataset.get_single_element())

  @combinations.generate(test_base.default_test_combinations())
  def testWindow(self):
    """Test that `get_single_element()` can consume a nested dataset."""

    def flat_map_func(ds):
      batched = ds.batch(2)
      element = batched.get_single_element()
      return dataset_ops.Dataset.from_tensors(element)

    dataset = dataset_ops.Dataset.range(10).window(2).flat_map(flat_map_func)
    self.assertDatasetProduces(dataset,
                               [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])

  @combinations.generate(test_base.default_test_combinations())
  def testSideEffect(self):
    counter_var = variables.Variable(0)

    def increment_fn(x):
      counter_var.assign_add(1)
      return x

    def dataset_fn():
      return dataset_ops.Dataset.range(1).map(increment_fn)

    @def_function.function
    def fn():
      _ = dataset_fn().get_single_element()
      return "hello"

    self.evaluate(counter_var.initializer)
    self.assertEqual(self.evaluate(fn()), b"hello")
    self.assertEqual(self.evaluate(counter_var), 1)

  @combinations.generate(test_base.default_test_combinations())
  def testAutomaticControlDependencies(self):
    counter_var = variables.Variable(1)

    def increment_fn(x):
      counter_var.assign(counter_var + 1)
      return x

    def multiply_fn(x):
      counter_var.assign(counter_var * 2)
      return x

    def dataset1_fn():
      return dataset_ops.Dataset.range(1).map(increment_fn)

    def dataset2_fn():
      return dataset_ops.Dataset.range(1).map(multiply_fn)

    @def_function.function
    def fn():
      _ = dataset1_fn().get_single_element()
      _ = dataset2_fn().get_single_element()
      return "hello"

    self.evaluate(counter_var.initializer)
    self.assertEqual(self.evaluate(fn()), b"hello")
    self.assertEqual(self.evaluate(counter_var), 4)

  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    dataset = dataset_ops.Dataset.from_tensors(42)
    self.assertEqual(
        self.evaluate(dataset.get_single_element(name="get_single_element")),
        42)


if __name__ == "__main__":
  test.main()
