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
"""Datasets for random number generators."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import sparse
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import gen_dataset_ops


class RandomDataset(dataset_ops.Dataset):
  """A `Dataset` of pseudorandom values."""

  def __init__(self, seed=None):
    """A `Dataset` of pseudorandom values."""
    super(RandomDataset, self).__init__()
    seed, seed2 = random_seed.get_seed(seed)
    if seed is None:
      self._seed = constant_op.constant(0, dtype=dtypes.int64, name="seed")
    else:
      self._seed = ops.convert_to_tensor(seed, dtype=dtypes.int64, name="seed")
    if seed2 is None:
      self._seed2 = constant_op.constant(0, dtype=dtypes.int64, name="seed2")
    else:
      self._seed2 = ops.convert_to_tensor(
          seed2, dtype=dtypes.int64, name="seed2")

  def _as_variant_tensor(self):
    return gen_dataset_ops.random_dataset(
        seed=self._seed,
        seed2=self._seed2,
        output_shapes=nest.flatten(
            sparse.as_dense_shapes(self.output_shapes, self.output_classes)),
        output_types=nest.flatten(
            sparse.as_dense_types(self.output_types, self.output_classes)))

  @property
  def output_classes(self):
    return ops.Tensor

  @property
  def output_shapes(self):
    return tensor_shape.scalar()

  @property
  def output_types(self):
    return dtypes.int64
