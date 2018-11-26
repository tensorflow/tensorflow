# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Test utilities for tf.data functionality."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class DatasetTestBase(test.TestCase):
  """Base class for dataset tests."""

  def assertSparseValuesEqual(self, a, b):
    """Asserts that two SparseTensors/SparseTensorValues are equal."""
    self.assertAllEqual(a.indices, b.indices)
    self.assertAllEqual(a.values, b.values)
    self.assertAllEqual(a.dense_shape, b.dense_shape)

  def getNext(self, dataset, requires_initialization=False):
    """Returns a callable that returns the next element of the dataset.

    Example use:
    ```python
    # In both graph and eager modes
    dataset = ...
    nxt = self.getNext(dataset)
    result = self.evaluate(nxt())
    ```

    Args:
      dataset: A dataset whose elements will be returned.
      requires_initialization: Indicates that when the test is executed in graph
        mode, it should use an initializable iterator to iterate through the
        dataset (e.g. when it contains stateful nodes). Defaults to False.
    Returns:
      A callable that returns the next element of `dataset`.
    """
    if context.executing_eagerly():
      iterator = dataset.__iter__()
      return iterator._next_internal  # pylint: disable=protected-access
    else:
      if requires_initialization:
        iterator = dataset.make_initializable_iterator()
        self.evaluate(iterator.initializer)
      else:
        iterator = dataset.make_one_shot_iterator()
      return iterator.get_next

  def _compareOutputToExpected(self, result_values, expected_values,
                               assert_items_equal):
    if assert_items_equal:
      # TODO(shivaniagrawal): add support for nested elements containing sparse
      # tensors when needed.
      self.assertItemsEqual(result_values, expected_values)
      return
    for i in range(len(result_values)):
      nest.assert_same_structure(result_values[i], expected_values[i])
      for result_value, expected_value in zip(
          nest.flatten(result_values[i]), nest.flatten(expected_values[i])):
        if sparse_tensor.is_sparse(result_value):
          self.assertSparseValuesEqual(result_value, expected_value)
        else:
          self.assertAllEqual(result_value, expected_value)

  def assertDatasetProduces(self,
                            dataset,
                            expected_output=None,
                            expected_error=None,
                            requires_initialization=False,
                            num_test_iterations=1,
                            assert_items_equal=False):
    """Asserts that a dataset produces the expected output / error.

    Args:
      dataset: A dataset to check for the expected output / error.
      expected_output: A list of elements that the dataset is expected to
        produce.
      expected_error: A tuple `(type, predicate)` identifying the expected error
        `dataset` should raise. The `type` should match the expected exception
        type, while `predicate` should either be 1) a unary function that inputs
        the raised exception and returns a boolean indicator of success or 2) a
        regular expression that is expected to match the error message
        partially.
      requires_initialization: Indicates that when the test is executed in graph
        mode, it should use an initializable iterator to iterate through the
        dataset (e.g. when it contains stateful nodes). Defaults to False.
      num_test_iterations: Number of times `dataset` will be iterated. Defaults
        to 2.
      assert_items_equal: Tests expected_output has (only) the same elements
        regardless of order.
    """
    self.assertTrue(
        expected_error is not None or expected_output is not None,
        "Exactly one of expected_output or expected error should be provided.")
    if expected_error:
      self.assertTrue(
          expected_output is None,
          "Exactly one of expected_output or expected error should be provided."
      )
      with self.assertRaisesWithPredicateMatch(expected_error[0],
                                               expected_error[1]):
        get_next = self.getNext(
            dataset, requires_initialization=requires_initialization)
        self.evaluate(get_next())
      return
    self.assertGreater(num_test_iterations, 0)
    for _ in range(num_test_iterations):
      get_next = self.getNext(
          dataset, requires_initialization=requires_initialization)
      result = []
      for _ in range(len(expected_output)):
        result.append(self.evaluate(get_next()))
      self._compareOutputToExpected(result, expected_output, assert_items_equal)
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(get_next())
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(get_next())

  def assertDatasetsEqual(self, dataset1, dataset2):
    """Checks that datasets are equal. Supports both graph and eager mode."""
    self.assertEqual(dataset1.output_types, dataset2.output_types)
    self.assertEqual(dataset1.output_classes, dataset2.output_classes)
    flattened_types = nest.flatten(dataset1.output_types)

    next1 = self.getNext(dataset1)
    next2 = self.getNext(dataset2)
    while True:
      try:
        op1 = self.evaluate(next1())
      except errors.OutOfRangeError:
        with self.assertRaises(errors.OutOfRangeError):
          self.evaluate(next2())
        break
      op2 = self.evaluate(next2())

      op1 = nest.flatten(op1)
      op2 = nest.flatten(op2)
      assert len(op1) == len(op2)
      for i in range(len(op1)):
        if sparse_tensor.is_sparse(op1[i]):
          self.assertSparseValuesEqual(op1[i], op2[i])
        elif flattened_types[i] == dtypes.string:
          self.assertAllEqual(op1[i], op2[i])
        else:
          self.assertAllClose(op1[i], op2[i])

  def assertDatasetsRaiseSameError(self,
                                   dataset1,
                                   dataset2,
                                   exception_class,
                                   replacements=None):
    """Checks that datasets raise the same error on the first get_next call."""
    next1 = self.getNext(dataset1)
    next2 = self.getNext(dataset2)
    try:
      self.evaluate(next1())
      raise ValueError(
          "Expected dataset to raise an error of type %s, but it did not." %
          repr(exception_class))
    except exception_class as e:
      expected_message = e.message
      for old, new, count in replacements:
        expected_message = expected_message.replace(old, new, count)
      # Check that the first segment of the error messages are the same.
      with self.assertRaisesRegexp(exception_class,
                                   re.escape(expected_message)):
        self.evaluate(next2())

  def structuredDataset(self, structure, shape=None, dtype=dtypes.int64):
    """Returns a singleton dataset with the given structure."""
    if shape is None:
      shape = []
    if structure is None:
      return dataset_ops.Dataset.from_tensors(
          array_ops.zeros(shape, dtype=dtype))
    else:
      return dataset_ops.Dataset.zip(
          tuple([
              self.structuredDataset(substructure, shape, dtype)
              for substructure in structure
          ]))

  def structuredElement(self, structure, shape=None, dtype=dtypes.int64):
    """Returns an element with the given structure."""
    if shape is None:
      shape = []
    if structure is None:
      return array_ops.zeros(shape, dtype=dtype)
    else:
      return tuple([
          self.structuredElement(substructure, shape, dtype)
          for substructure in structure
      ])
