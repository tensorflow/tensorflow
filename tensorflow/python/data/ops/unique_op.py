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
"""The implementation of `tf.data.Dataset.unique`."""

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_experimental_dataset_ops


def _unique(input_dataset, name):  # pylint: disable=unused-private-name
  return _UniqueDataset(input_dataset, name)


class _UniqueDataset(dataset_ops.UnaryUnchangedStructureDataset):
  """A dataset containing the unique elements of an input dataset."""

  def __init__(self, input_dataset, name=None):
    """See `tf.data.Dataset.unique` for details."""
    self._input_dataset = input_dataset
    for ty in nest.flatten(dataset_ops.get_legacy_output_types(input_dataset)):
      if ty not in (dtypes.int32, dtypes.int64, dtypes.string):
        raise TypeError(
            f"`tf.data.Dataset.unique` does not support type {ty} -- only "
            f"`tf.int32`, `tf.int64`, and `tf.string` are supported.")
    self._name = name
    variant_tensor = gen_experimental_dataset_ops.unique_dataset(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
        **self._common_args)
    super().__init__(input_dataset, variant_tensor)
