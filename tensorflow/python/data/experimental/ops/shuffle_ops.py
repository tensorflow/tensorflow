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
"""Experimental shuffle ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import random_seed
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


class _ShuffleAndRepeatDataset(dataset_ops.UnaryUnchangedStructureDataset):
  """A `Dataset` that fuses `shuffle` and `repeat`."""

  def __init__(self, input_dataset, buffer_size, count=None, seed=None):
    self._input_dataset = input_dataset
    self._buffer_size = ops.convert_to_tensor(
        buffer_size, dtype=dtypes.int64, name="buffer_size")
    if count is None:
      self._count = constant_op.constant(-1, dtype=dtypes.int64, name="count")
    else:
      self._count = ops.convert_to_tensor(
          count, dtype=dtypes.int64, name="count")
    self._seed, self._seed2 = random_seed.get_seed(seed)
    variant_tensor = gen_dataset_ops.shuffle_and_repeat_dataset(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
        buffer_size=self._buffer_size,
        count=self._count,
        seed=self._seed,
        seed2=self._seed2,
        **self._flat_structure)
    super(_ShuffleAndRepeatDataset, self).__init__(input_dataset,
                                                   variant_tensor)


@deprecation.deprecated(
    None,
    "Use `tf.data.Dataset.shuffle(buffer_size, seed)` followed by "
    "`tf.data.Dataset.repeat(count)`. Static tf.data optimizations will take "
    "care of using the fused implementation.")
@tf_export("data.experimental.shuffle_and_repeat")
def shuffle_and_repeat(buffer_size, count=None, seed=None):
  """Shuffles and repeats a Dataset, reshuffling with each repetition.

  >>> d = tf.data.Dataset.from_tensor_slices([1, 2, 3])
  >>> d = d.apply(tf.data.experimental.shuffle_and_repeat(2, count=2))
  >>> [elem.numpy() for elem in d] # doctest: +SKIP
  [2, 3, 1, 1, 3, 2]

  ```python
  dataset.apply(
    tf.data.experimental.shuffle_and_repeat(buffer_size, count, seed))
  ```

  produces the same output as

  ```python
  dataset.shuffle(
    buffer_size, seed=seed, reshuffle_each_iteration=True).repeat(count)
  ```

  In each repetition, this dataset fills a buffer with `buffer_size` elements,
  then randomly samples elements from this buffer, replacing the selected
  elements with new elements. For perfect shuffling, set the buffer size equal
  to the full size of the dataset.

  For instance, if your dataset contains 10,000 elements but `buffer_size` is
  set to 1,000, then `shuffle` will initially select a random element from
  only the first 1,000 elements in the buffer. Once an element is selected,
  its space in the buffer is replaced by the next (i.e. 1,001-st) element,
  maintaining the 1,000 element buffer.

  Args:
    buffer_size: A `tf.int64` scalar `tf.Tensor`, representing the maximum
      number elements that will be buffered when prefetching.
    count: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the number
      of times the dataset should be repeated. The default behavior (if `count`
      is `None` or `-1`) is for the dataset be repeated indefinitely.
    seed: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the random
      seed that will be used to create the distribution. See
      `tf.random.set_seed` for behavior.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

  def _apply_fn(dataset):  # pylint: disable=missing-docstring
    return _ShuffleAndRepeatDataset(dataset, buffer_size, count, seed)

  return _apply_fn
