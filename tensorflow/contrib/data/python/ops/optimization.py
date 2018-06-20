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
"""Experimental API for optimizing `tf.data` pipelines."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.data.python.ops import contrib_op_loader  # pylint: disable=unused-import
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops


def optimize(optimizations=None):
  """A transformation that applies optimizations.

  Args:
    optimizations: (Optional.) A `tf.string` vector `tf.Tensor` identifying
      optimizations to use. If not specified, the default set of optimizations
      is applied.

  Returns:
    A `Dataset` transformation function, which can be passed to
    @{tf.data.Dataset.apply}.
  """

  def _apply_fn(dataset):
    """Function from `Dataset` to `Dataset` that applies the transformation."""
    return OptimizeDataset(dataset, optimizations)

  return _apply_fn


class OptimizeDataset(dataset_ops.Dataset):
  """A `Dataset` that acts as an identity, and applies optimizations."""

  def __init__(self, input_dataset, optimizations):
    """See `optimize()` for details."""
    super(OptimizeDataset, self).__init__()
    self._input_dataset = input_dataset
    if optimizations is None:
      optimizations = []
    self._optimizations = ops.convert_to_tensor(
        optimizations, dtype=dtypes.string, name="optimizations")

  def _as_variant_tensor(self):
    return gen_dataset_ops.optimize_dataset(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
        self._optimizations,
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
