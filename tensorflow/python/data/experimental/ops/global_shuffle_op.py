# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
"""Globally shuffles tf.data datasets."""

from typing import Optional, Union

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.util import random_seed
from tensorflow.python.framework import tensor
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops


def _global_shuffle(  # pylint: disable=unused-private-name
    input_dataset: dataset_ops.DatasetV2,
    seed: Optional[Union[int, tensor.Tensor]] = None,
    reshuffle_each_iteration: bool = True,
    name: Optional[str] = None) -> dataset_ops.DatasetV2:
  """Globally shuffles the elements of `input_dataset`.

  The shuffling is done efficiently, without needing to buffer any additional
  data. To achieve this, the transformations preceding global_shuffle must all
  support random access.

  Requires that:
  - The shuffled dataset and all its input datasets support random access.
  - The input_dataset to have a known, finite cardinality. Users can use
    `tf.data.experimental.assert_cardinality` to specify the cardinality of a
    dataset if it cannot be determined at runtime.

  TODO(b/325112575): Move the API to dataset_ops.py.

  Args:
    input_dataset: The dataset to be shuffled.
    seed: An int or `tf.int64` scalar `tf.Tensor` to control the shuffle order.
      If `None`, a random seed will be used.
    reshuffle_each_iteration: A boolean, which if True, indicates that a
      different shuffle order should be generated for each iteration of the
      dataset. (Defaults to `True`.)
    name: (Optional.) A name for the tf.data operation.

  Returns:
    A new `Dataset` where elements are produced in a globally shuffled order.

  Raises:
    - InvalidArgumentError if the input dataset does not support random access,
      or it has infinite or unknown cardinality.
    - FailedPreconditionError for batching with `drop_remainder=False`.
  """
  return _GlobalShuffleDataset(
      input_dataset,
      seed=seed,
      reshuffle_each_iteration=reshuffle_each_iteration,
      name=name)


class _GlobalShuffleDataset(dataset_ops.UnaryUnchangedStructureDataset):
  """Shuffles all elements in the input dataset."""

  def __init__(
      self,
      input_dataset: dataset_ops.DatasetV2,
      seed: Optional[Union[int, tensor.Tensor]] = None,
      reshuffle_each_iteration: bool = True,
      name: Optional[str] = None):

    options = options_lib.Options()
    # Currently, prefetching threads cannot access the runtime context required
    # for global shuffling when `warm_start` is enabled. Supporting it will be
    # future work.
    options.experimental_warm_start = False
    input_dataset = input_dataset.with_options(options)

    self._input_dataset = input_dataset
    self._seed, self._seed2 = random_seed.get_seed(seed)
    self._reshuffle_each_iteration = reshuffle_each_iteration
    self._name = name
    variant_tensor = ged_ops.global_shuffle_dataset(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
        seed=self._seed,
        seed2=self._seed2,
        seed_generator=gen_dataset_ops.dummy_seed_generator(),
        reshuffle_each_iteration=self._reshuffle_each_iteration,
        **self._common_args)
    super().__init__(input_dataset, variant_tensor)
