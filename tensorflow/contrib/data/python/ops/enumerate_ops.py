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
"""Enumerate dataset transformations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_dataset_ops


def enumerate_dataset(start=0):
  """A transformation that enumerate the elements of a dataset.

  It is Similar to python's `enumerate`.
  For example:

  ```python
  # NOTE: The following examples use `{ ... }` to represent the
  # contents of a dataset.
  a = { 1, 2, 3 }
  b = { (7, 8), (9, 10) }

  # The nested structure of the `datasets` argument determines the
  # structure of elements in the resulting dataset.
  a.apply(tf.contrib.data.enumerate(start=5)) == { (5, 1), (6, 2), (7, 3) }
  b.apply(tf.contrib.data.enumerate()) == { (0, (7, 8)), (1, (9, 10)) }
  ```

  Args:
    start: A `tf.int64` scalar `tf.Tensor`, representing the start
      value for enumeration.

  Returns:
    A `Dataset` transformation function, which can be passed to
    @{tf.contrib.data.Dataset.apply}.
  """

  def _apply_fn(dataset):
    max_value = np.iinfo(dtypes.int64.as_numpy_dtype).max
    return dataset_ops.Dataset.zip((dataset_ops.Dataset.range(start, max_value),
                                    dataset))

  return _apply_fn


def ignore_errors():
  """Creates a `Dataset` from another `Dataset` and silently ignores any errors.

  Use this transformation to produce a dataset that contains the same elements
  as the input, but silently drops any elements that caused an error. For
  example:

  ```python
  dataset = tf.contrib.data.Dataset.from_tensor_slices([1., 2., 0., 4.])

  # Computing `tf.check_numerics(1. / 0.)` will raise an InvalidArgumentError.
  dataset = dataset.map(lambda x: tf.check_numerics(1. / x, "error"))

  # Using `ignore_errors()` will drop the element that causes an error.
  dataset =
      dataset.apply(tf.contrib.data.ignore_errors())  # ==> { 1., 0.5, 0.2 }
  ```

  Returns:
    A `Dataset` transformation function, which can be passed to
    @{tf.contrib.data.Dataset.apply}.
  """

  def _apply_fn(dataset):
    return IgnoreErrorsDataset(dataset)

  return _apply_fn


class IgnoreErrorsDataset(dataset_ops.Dataset):
  """A `Dataset` that silently ignores errors when computing its input."""

  def __init__(self, input_dataset):
    """See `Dataset.ignore_errors()` for details."""
    super(IgnoreErrorsDataset, self).__init__()
    self._input_dataset = input_dataset

  def make_dataset_resource(self):
    return gen_dataset_ops.ignore_errors_dataset(
        self._input_dataset.make_dataset_resource(),
        output_shapes=nest.flatten(self.output_shapes),
        output_types=nest.flatten(self.output_types))

  @property
  def output_shapes(self):
    return self._input_dataset.output_shapes

  @property
  def output_types(self):
    return self._input_dataset.output_types
