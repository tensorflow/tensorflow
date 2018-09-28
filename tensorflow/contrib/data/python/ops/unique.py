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
"""Unique element dataset transformations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_experimental_dataset_ops


def unique():
  """Creates a `Dataset` from another `Dataset`, discarding duplicates.

  Use this transformation to produce a dataset that contains one instance of
  each unique element in the input. For example:

  ```python
  dataset = tf.data.Dataset.from_tensor_slices([1, 37, 2, 37, 2, 1])

  # Using `unique()` will drop the duplicate elements.
  dataset = dataset.apply(tf.contrib.data.unique())  # ==> { 1, 37, 2 }
  ```

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

  def _apply_fn(dataset):
    return _UniqueDataset(dataset)

  return _apply_fn


class _UniqueDataset(dataset_ops.UnaryDataset):
  """A `Dataset` contains the unique elements from its input."""

  def __init__(self, input_dataset):
    """See `unique()` for details."""
    super(_UniqueDataset, self).__init__(input_dataset)
    self._input_dataset = input_dataset
    if input_dataset.output_types not in (dtypes.int32, dtypes.int64,
                                          dtypes.string):
      raise TypeError(
          "`tf.contrib.data.unique()` only supports inputs with a single "
          "`tf.int32`, `tf.int64`, or `tf.string` component.")

  def _as_variant_tensor(self):
    return gen_experimental_dataset_ops.experimental_unique_dataset(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
        **dataset_ops.flat_structure(self))

  @property
  def output_classes(self):
    return self._input_dataset.output_classes

  @property
  def output_shapes(self):
    return self._input_dataset.output_shapes

  @property
  def output_types(self):
    return self._input_dataset.output_types
