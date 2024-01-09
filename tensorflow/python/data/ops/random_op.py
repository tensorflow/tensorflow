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
"""The implementation of `tf.data.Dataset.random`."""

import warnings

from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import random_seed
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops


def _random(  # pylint: disable=unused-private-name
    seed=None,
    rerandomize_each_iteration=None,
    name=None):
  """See `Dataset.random()` for details."""
  return _RandomDataset(
      seed=seed,
      rerandomize_each_iteration=rerandomize_each_iteration,
      name=name)


class _RandomDataset(dataset_ops.DatasetSource):
  """A `Dataset` of pseudorandom values."""

  def __init__(self, seed=None, rerandomize_each_iteration=None, name=None):
    """A `Dataset` of pseudorandom values."""
    self._seed, self._seed2 = random_seed.get_seed(seed)
    self._rerandomize = rerandomize_each_iteration
    self._name = name
    if rerandomize_each_iteration:
      if not tf2.enabled():
        warnings.warn("In TF 1, the `rerandomize_each_iteration=True` option "
                      "is only supported for repeat-based epochs.")
      variant_tensor = ged_ops.random_dataset_v2(
          seed=self._seed,
          seed2=self._seed2,
          seed_generator=gen_dataset_ops.dummy_seed_generator(),
          rerandomize_each_iteration=self._rerandomize,
          **self._common_args)
    else:
      variant_tensor = ged_ops.random_dataset(
          seed=self._seed, seed2=self._seed2, **self._common_args)
    super().__init__(variant_tensor)

  @property
  def element_spec(self):
    return tensor_spec.TensorSpec([], dtypes.int64)
