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

  def getNext(self, dataset):
    """Returns a callable that returns the next element of the dataset.

    Example use:
    ```python
    # In both graph and eager modes
    dataset = ...
    nxt = self.getNext(dataset)
    result = self.evaluate(nxt())
    ```

    Args:
      dataset: A dataset whose next element is returned

    Returns:
      A callable that returns the next element of `dataset`
    """
    it = dataset.make_one_shot_iterator()
    if context.executing_eagerly():
      return it.get_next
    else:
      nxt = it.get_next()
      return lambda: nxt

  def _compare_output_to_expected(self, result_values, expected_values):
    for i in range(len(result_values)):
      if sparse_tensor.is_sparse(result_values[i]):
        self.assertSparseValuesEqual(result_values[i], expected_values[i])
      else:
        self.assertAllEqual(result_values[i], expected_values[i])

  def assertDatasetProduces(self,
                            input_dataset,
                            expected_output=None,
                            expected_err=None,
                            create_iterator_twice=True):

    if expected_err:
      with self.assertRaisesWithPredicateMatch(expected_err[0],
                                               expected_err[1]):
        get_next = self.getNext(input_dataset)
        self.evaluate(get_next())
      return
    repeated = 2 if create_iterator_twice else 1
    for _ in range(repeated):
      get_next = self.getNext(input_dataset)
      result = []
      for _ in range(len(expected_output)):
        result.append(self.evaluate(get_next()))
      self._compare_output_to_expected(result, expected_output)
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
          'Expected dataset to raise an error of type %s, but it did not.' %
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
