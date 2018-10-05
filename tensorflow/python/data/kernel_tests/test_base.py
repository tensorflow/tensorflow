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

from tensorflow.python.data.util import nest
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
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

  def assertDatasetsEqual(self, dataset1, dataset2):
    """Checks that datasets are equal. Supports both graph and eager mode."""
    self.assertEqual(dataset1.output_types, dataset2.output_types)
    self.assertEqual(dataset1.output_classes, dataset2.output_classes)

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
        if isinstance(
            op1[i],
            (sparse_tensor.SparseTensor, sparse_tensor.SparseTensorValue)):
          self.assertSparseValuesEqual(op1[i], op2[i])
        else:
          self.assertAllEqual(op1[i], op2[i])

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
