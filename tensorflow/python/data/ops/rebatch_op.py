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
"""The implementation of `tf.data.Dataset.rebatch`."""

import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops


def _rebatch(input_dataset, batch_size, drop_remainder=False, name=None):
  return _RebatchDataset(input_dataset, batch_size, drop_remainder, name)


class _RebatchDataset(dataset_ops.UnaryDataset):
  """A `Dataset` that rebatches elements from its input into new batch sizes.

  `_RebatchDataset(input_dataset, batch_sizes)` is functionally equivalent to
  `input_dataset.unbatch().batch(N)`, where the value of N cycles through the
  `batch_sizes` input list. The elements produced by this dataset have the same
  rank as the elements of the input dataset.
  """

  def __init__(self,
               input_dataset,
               batch_sizes,
               drop_remainder=False,
               name=None):
    """See `Dataset.rebatch` for details."""
    self._input_dataset = input_dataset
    self._batch_sizes = ops.convert_to_tensor(
        batch_sizes, dtype=dtypes.int64, name="batch_sizes")
    self._drop_remainder = ops.convert_to_tensor(
        drop_remainder, dtype=dtypes.bool, name="drop_remainder")
    self._name = name
    new_batch_dim = self._compute_static_batch_dim()

    # pylint: disable=protected-access
    self._element_spec = nest.map_structure(
        lambda ts: ts._unbatch()._batch(new_batch_dim),
        dataset_ops.get_structure(input_dataset))
    # pylint: enable=protected-access

    # auto_shard rewrite assumes that there's normalize_to_dense before
    # rebatch_dataset.
    # LINT.IfChange
    input_dataset = dataset_ops.normalize_to_dense(input_dataset)
    variant_tensor = ged_ops.rebatch_dataset_v2(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        batch_sizes=batch_sizes,
        drop_remainder=drop_remainder,
        **self._flat_structure)
    # LINT.ThenChange(//tensorflow/core/grappler/optimizers/data/auto_shard.cc)
    super().__init__(input_dataset, variant_tensor)

  def _compute_static_batch_dim(self):
    """Computes the static batch dimension of a dataset if it can be determined.

    Given the RebatchDataset parameters, determines the batch dimension of this
    dataset statically. Returns None if this cannot be determined or is
    variable.

    Returns:
      An integer representing the batch dimension of the dataset. If it cannot
      be determined statically, returns None.

    Raises:
      ValueError: The batch_sizes parameter is malformed, input_dataset is
      not batched, or input_dataset batch sizes are incompatible with each
      other.
    """
    new_batch_dim = tensor_util.constant_value(self._batch_sizes)
    if new_batch_dim is None:
      return None

    if isinstance(new_batch_dim, np.ndarray):
      if len(new_batch_dim.shape) == 1:
        if np.all(new_batch_dim == new_batch_dim[0]):
          new_batch_dim = new_batch_dim[0]
        else:
          return None
      elif len(new_batch_dim.shape) > 1:
        raise ValueError(
            f"Invalid `batch_sizes`. Expected `batch_sizes` to be a scalar or "
            f"a vector. Received `batch_sizes` of rank "
            f"{len(new_batch_dim.shape)}.")

    if self._may_form_partial_batches(new_batch_dim):
      return None

    return new_batch_dim

  def _may_form_partial_batches(self, desired_batch_size):
    """Returns whether this dataset may form partial batches."""
    if tensor_util.constant_value(self._drop_remainder):
      return False

    def get_batch_dim(type_spec):
      try:
        shape = type_spec._to_legacy_output_shapes()  # pylint: disable=protected-access
      except NotImplementedError:
        return None
      if not isinstance(shape, tensor_shape.TensorShape):
        return None
      if shape.rank is None:
        return None
      if len(shape) < 1:
        raise ValueError("Invalid `batch_sizes`. Expected dataset with "
                         "rank of >= 1 but found a dataset with "
                         "scalar elements. Fix the issue by adding the `batch` "
                         "transformation to the dataset.")
      return shape.dims[0].value

    input_batch_dims = [
        get_batch_dim(ts)
        for ts in nest.flatten(dataset_ops.get_structure(self._input_dataset))
    ]
    known_input_batch_dims = [d for d in input_batch_dims if d is not None]

    if not known_input_batch_dims:
      return True

    known_input_batch_dims = np.asarray(known_input_batch_dims)
    if not np.all(known_input_batch_dims == known_input_batch_dims[0]):
      raise ValueError(
          f"Invalid `input_dataset.` The batch dimension of component 0 "
          f"is {known_input_batch_dims[0]}, while the batch dimension "
          f"of component i is {known_input_batch_dims}.")

    return known_input_batch_dims[0] % desired_batch_size != 0

  @property
  def element_spec(self):
    return self._element_spec
