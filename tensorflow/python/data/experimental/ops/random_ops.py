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

import functools

from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import random_seed
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.util.tf_export import tf_export


@tf_export("data.experimental.RandomDataset", v1=[])
class RandomDatasetV2(dataset_ops.DatasetSource):
  """A `Dataset` of pseudorandom values."""

  def __init__(self, seed=None):
    """A `Dataset` of pseudorandom values."""
    self._seed, self._seed2 = random_seed.get_seed(seed)
    variant_tensor = gen_experimental_dataset_ops.random_dataset(
        seed=self._seed, seed2=self._seed2, **self._flat_structure)
    super(RandomDatasetV2, self).__init__(variant_tensor)

  @property
  def element_spec(self):
    return tensor_spec.TensorSpec([], dtypes.int64)


@tf_export(v1=["data.experimental.RandomDataset"])
class RandomDatasetV1(dataset_ops.DatasetV1Adapter):
  """A `Dataset` of pseudorandom values."""

  @functools.wraps(RandomDatasetV2.__init__)
  def __init__(self, seed=None):
    wrapped = RandomDatasetV2(seed)
    super(RandomDatasetV1, self).__init__(wrapped)


if tf2.enabled():
  RandomDataset = RandomDatasetV2
else:
  RandomDataset = RandomDatasetV1
