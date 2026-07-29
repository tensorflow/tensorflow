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
"""The implementation of `tf.data.Dataset.shuffle`."""
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import random_seed
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops

# Sanity limit on the number of elements in the shuffle buffer. The C++
# kernel eagerly allocates a slot for every element up to `buffer_size` when
# the iterator is created, so a pathologically large value (e.g.
# `sys.maxsize`) reaches that allocation and crashes the process instead of
# raising a catchable error.
_MAX_SHUFFLE_BUFFER_SIZE_ELEMENTS = 1 << 30  # ~1 billion elements


def _shuffle(  # pylint: disable=unused-private-name
    input_dataset,
    buffer_size,
    seed=None,
    reshuffle_each_iteration=True,
    name=None,
):
  return _ShuffleDataset(
      input_dataset, buffer_size, seed, reshuffle_each_iteration, name=name)


class _ShuffleDataset(dataset_ops.UnaryUnchangedStructureDataset):
  """A `Dataset` that randomly shuffles the elements of its input."""

  def __init__(
      self,
      input_dataset,
      buffer_size,
      seed=None,
      reshuffle_each_iteration=True,
      name=None,
  ):
    """See `Dataset.shuffle()` for details."""
    if (isinstance(buffer_size, int) and
        buffer_size > _MAX_SHUFFLE_BUFFER_SIZE_ELEMENTS):
      raise ValueError(
          f"`buffer_size` must not exceed "
          f"{_MAX_SHUFFLE_BUFFER_SIZE_ELEMENTS} elements, but got "
          f"{buffer_size}. Requesting a shuffle buffer this large would "
          "cause the dataset to abort the process instead of raising a "
          "catchable error.")
    self._input_dataset = input_dataset
    self._buffer_size = ops.convert_to_tensor(
        buffer_size, dtype=dtypes.int64, name="buffer_size")
    self._seed, self._seed2 = random_seed.get_seed(seed)
    self._reshuffle_each_iteration = reshuffle_each_iteration
    self._name = name

    if (tf2.enabled() and
        (context.executing_eagerly() or ops.inside_function())):
      variant_tensor = gen_dataset_ops.shuffle_dataset_v3(
          input_dataset._variant_tensor,  # pylint: disable=protected-access
          buffer_size=self._buffer_size,
          seed=self._seed,
          seed2=self._seed2,
          seed_generator=gen_dataset_ops.dummy_seed_generator(),
          reshuffle_each_iteration=self._reshuffle_each_iteration,
          **self._common_args)
    else:
      variant_tensor = gen_dataset_ops.shuffle_dataset(
          input_dataset._variant_tensor,  # pylint: disable=protected-access
          buffer_size=self._buffer_size,
          seed=self._seed,
          seed2=self._seed2,
          reshuffle_each_iteration=self._reshuffle_each_iteration,
          **self._common_args)
    super().__init__(input_dataset, variant_tensor)
