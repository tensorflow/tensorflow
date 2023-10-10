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
"""The implementation of `tf.data.Dataset.unbatch`."""

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops


def _unbatch(input_dataset, name=None):  # pylint: disable=unused-private-name
  """See `Dataset.unbatch()` for details."""
  normalized_dataset = dataset_ops.normalize_to_dense(input_dataset)
  return _UnbatchDataset(normalized_dataset, name=name)


class _UnbatchDataset(dataset_ops.UnaryDataset):
  """A dataset that splits the elements of its input into multiple elements."""

  def __init__(self, input_dataset, name=None):
    """See `unbatch()` for more details."""
    flat_shapes = input_dataset._flat_shapes  # pylint: disable=protected-access
    if any(s.ndims == 0 for s in flat_shapes):
      raise ValueError("Cannot unbatch an input with scalar components.")
    known_batch_dim = tensor_shape.Dimension(None)
    for s in flat_shapes:
      try:
        known_batch_dim = known_batch_dim.merge_with(s[0])
      except ValueError as e:
        raise ValueError(
            f"`unbatch()` is only supported for datasets of elements whose "
            f"components have a matching leading dimension. Encountered both "
            f"{known_batch_dim} and {s[0]}.") from e
    self._input_dataset = input_dataset
    self._structure = nest.map_structure(
        lambda component_spec: component_spec._unbatch(),  # pylint: disable=protected-access
        dataset_ops.get_structure(input_dataset))
    self._name = name
    variant_tensor = ged_ops.unbatch_dataset(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
        **self._common_args)
    super().__init__(input_dataset, variant_tensor)

  @property
  def element_spec(self):
    return self._structure
