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
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test


def default_test_combinations():
  """Returns the default test combinations for tf.data tests."""
  return combinations.combine(tf_api_version=[1, 2], mode=["eager", "graph"])


def eager_only_combinations():
  """Returns the default test combinations for eager mode only tf.data tests."""
  return combinations.combine(tf_api_version=[1, 2], mode="eager")


def graph_only_combinations():
  """Returns the default test combinations for graph mode only tf.data tests."""
  return combinations.combine(tf_api_version=[1, 2], mode="graph")


def v1_only_combinations():
  """Returns the default test combinations for v1 only tf.data tests."""
  return combinations.combine(tf_api_version=1, mode=["eager", "graph"])


def v2_only_combinations():
  """Returns the default test combinations for v2 only tf.data tests."""
  return combinations.combine(tf_api_version=2, mode=["eager", "graph"])


def v2_eager_only_combinations():
  """Returns the default test combinations for v2 eager only tf.data tests."""
  return combinations.combine(tf_api_version=2, mode="eager")


class DatasetTestBase(test.TestCase):
  """Base class for dataset tests."""

  def assert_op_cancelled(self, op):
    with self.assertRaises(errors.CancelledError):
      self.evaluate(op)

  def assertValuesEqual(self, expected, actual):
    """Asserts that two values are equal."""
    if isinstance(expected, dict):
      self.assertItemsEqual(list(expected.keys()), list(actual.keys()))
      for k in expected.keys():
        self.assertValuesEqual(expected[k], actual[k])
    elif sparse_tensor.is_sparse(expected):
      self.assertAllEqual(expected.indices, actual.indices)
      self.assertAllEqual(expected.values, actual.values)
      self.assertAllEqual(expected.dense_shape, actual.dense_shape)
    else:
      self.assertAllEqual(expected, actual)

  def getNext(self, dataset, requires_initialization=False, shared_name=None):
    """Returns a callable that returns the next element of the dataset.

    Example use:
    ```python
    # In both graph and eager modes
    dataset = ...
    get_next = self.getNext(dataset)
    result = self.evaluate(get_next())
    ```

    Args:
      dataset: A dataset whose elements will be returned.
      requires_initialization: Indicates that when the test is executed in graph
        mode, it should use an initializable iterator to iterate through the
        dataset (e.g. when it contains stateful nodes). Defaults to False.
      shared_name: (Optional.) If non-empty, the returned iterator will be
        shared under the given name across multiple sessions that share the same
        devices (e.g. when using a remote server).
    Returns:
      A callable that returns the next element of `dataset`. Any `TensorArray`
      objects `dataset` outputs are stacked.
    """
    def ta_wrapper(gn):
      def _wrapper():
        r = gn()
        if isinstance(r, tensor_array_ops.TensorArray):
          return r.stack()
        else:
          return r
      return _wrapper

    # Create an anonymous iterator if we are in eager-mode or are graph inside
    # of a tf.function.
    if context.executing_eagerly() or ops.inside_function():
      iterator = iter(dataset)
      return ta_wrapper(iterator._next_internal)  # pylint: disable=protected-access
    else:
      if requires_initialization:
        iterator = dataset_ops.make_initializable_iterator(dataset, shared_name)
        self.evaluate(iterator.initializer)
      else:
        iterator = dataset_ops.make_one_shot_iterator(dataset)
      get_next = iterator.get_next()
      return ta_wrapper(lambda: get_next)

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
        self.assertValuesEqual(expected_value, result_value)

  def getDatasetOutput(self, dataset, requires_initialization=False):
    get_next = self.getNext(
        dataset, requires_initialization=requires_initialization)
    return self.getIteratorOutput(get_next)

  def getIteratorOutput(self, get_next):
    """Evaluates `get_next` until end of input, returning the results."""
    results = []
    while True:
      try:
        results.append(self.evaluate(get_next()))
      except errors.OutOfRangeError:
        break
    return results

  def assertDatasetProduces(self,
                            dataset,
                            expected_output=None,
                            expected_shapes=None,
                            expected_error=None,
                            requires_initialization=False,
                            num_test_iterations=1,
                            assert_items_equal=False,
                            expected_error_iter=1):
    """Asserts that a dataset produces the expected output / error.

    Args:
      dataset: A dataset to check for the expected output / error.
      expected_output: A list of elements that the dataset is expected to
        produce.
      expected_shapes: A list of TensorShapes which is expected to match
        output_shapes of dataset.
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
        to 1.
      assert_items_equal: Tests expected_output has (only) the same elements
        regardless of order.
      expected_error_iter: How many times to iterate before expecting an error,
        if an error is expected.
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
        for _ in range(expected_error_iter):
          self.evaluate(get_next())
      return
    if expected_shapes:
      self.assertEqual(expected_shapes,
                       dataset_ops.get_legacy_output_shapes(dataset))
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
    self.assertTrue(
        structure.are_compatible(
            dataset_ops.get_structure(dataset1),
            dataset_ops.get_structure(dataset2)))

    flattened_types = nest.flatten(
        dataset_ops.get_legacy_output_types(dataset1))

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
        if sparse_tensor.is_sparse(op1[i]) or ragged_tensor.is_ragged(op1[i]):
          self.assertValuesEqual(op1[i], op2[i])
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
    if replacements is None:
      replacements = []
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

  def structuredDataset(self, dataset_structure, shape=None,
                        dtype=dtypes.int64):
    """Returns a singleton dataset with the given structure."""
    if shape is None:
      shape = []
    if dataset_structure is None:
      return dataset_ops.Dataset.from_tensors(
          array_ops.zeros(shape, dtype=dtype))
    else:
      return dataset_ops.Dataset.zip(
          tuple([
              self.structuredDataset(substructure, shape, dtype)
              for substructure in dataset_structure
          ]))

  def graphRoundTrip(self, dataset, allow_stateful=False):
    """Converts a dataset to a graph and back."""
    graph = gen_dataset_ops.dataset_to_graph(
        dataset._variant_tensor, allow_stateful=allow_stateful)  # pylint: disable=protected-access
    return dataset_ops.from_variant(
        gen_experimental_dataset_ops.dataset_from_graph(graph),
        dataset.element_spec)

  def structuredElement(self, element_structure, shape=None,
                        dtype=dtypes.int64):
    """Returns an element with the given structure."""
    if shape is None:
      shape = []
    if element_structure is None:
      return array_ops.zeros(shape, dtype=dtype)
    else:
      return tuple([
          self.structuredElement(substructure, shape, dtype)
          for substructure in element_structure
      ])

  def checkDeterminism(self, dataset_fn, expect_determinism, expected_elements):
    """Tests whether a dataset produces its elements deterministically.

    `dataset_fn` takes a delay_ms argument, which tells it how long to delay
    production of the first dataset element. This gives us a way to trigger
    out-of-order production of dataset elements.

    Args:
      dataset_fn: A function taking a delay_ms argument.
      expect_determinism: Whether to expect deterministic ordering.
      expected_elements: The elements expected to be produced by the dataset,
        assuming the dataset produces elements in deterministic order.
    """
    if expect_determinism:
      dataset = dataset_fn(100)
      actual = self.getDatasetOutput(dataset)
      self.assertAllEqual(expected_elements, actual)
      return

    # We consider the test a success if it succeeds under any delay_ms. The
    # delay_ms needed to observe non-deterministic ordering varies across
    # test machines. Usually 10 or 100 milliseconds is enough, but on slow
    # machines it could take longer.
    for delay_ms in [10, 100, 1000, 20000]:
      dataset = dataset_fn(delay_ms)
      actual = self.getDatasetOutput(dataset)
      self.assertCountEqual(expected_elements, actual)
      for i in range(len(actual)):
        if actual[i] != expected_elements[i]:
          return
    self.fail("Failed to observe nondeterministic ordering")
