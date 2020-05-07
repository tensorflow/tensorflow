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
from tensorflow.python.util.tf_export import tf_export


@tf_export("data.experimental.unique")
def unique():
  """Creates a `Dataset` from another `Dataset`, discarding duplicates.

  Use this transformation to produce a dataset that contains one instance of
  each unique element in the input. For example:

  ```python
  dataset = tf.data.Dataset.from_tensor_slices([1, 37, 2, 37, 2, 1])

  # Using `unique()` will drop the duplicate elements.
  dataset = dataset.apply(tf.data.experimental.unique())  # ==> { 1, 37, 2 }
  ```

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

  def _apply_fn(dataset):
    return _UniqueDataset(dataset)

  return _apply_fn


class _UniqueDataset(dataset_ops.UnaryUnchangedStructureDataset):
  """A `Dataset` contains the unique elements from its input."""

  def __init__(self, input_dataset):
    """See `unique()` for details."""
    self._input_dataset = input_dataset
    if dataset_ops.get_legacy_output_types(input_dataset) not in (
        dtypes.int32, dtypes.int64, dtypes.string):
      raise TypeError(
          "`tf.data.experimental.unique()` only supports inputs with a single "
          "`tf.int32`, `tf.int64`, or `tf.string` component.")
    variant_tensor = gen_experimental_dataset_ops.unique_dataset(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
        **self._flat_structure)
    super(_UniqueDataset, self).__init__(input_dataset, variant_tensor)
