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

import functools
import numpy as np

from tensorflow.python.data.experimental.ops import random_access
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import random_seed
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import stateless_random_ops
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
    None, "Use `tf.data.Dataset.shuffle(buffer_size, seed)` followed by "
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


def _process_file_infos(file_infos):
  """Computes aggregate information about files to read.

  The method collects information about the files to read, the total number of
  elements, and arrays that can be used to account for elements to be skipped,
  which can be specified via the "skip" and "take" keys.

  To account for elements to skip, the range of each file can be divided into
  three regions:
  - S (elements to skip)
  - T (elements to read)
  - R (remainder of elements that will also be skipped)

  The `thresholds` and `offsets` arrays are initialized as follows:
  `thresholds = [0, T_1, T_1 + T_2, ...]` and
  `offsets = [S_1, S_1 + R_1 + S_2, S_1 + R_1 + S_2 + R_2 + S_3, ...]`

  This makes it possible to map an index from a contiguous range
  `(0...num_elements_to_read)` to an index in the range of all elements,
  skipping over elements as per the "skip" and "take" keys values. In
  particular, for a given input index `X`, we find the greatest `thresholds`
  value that is smaller or equal to `X`. Let `t(X)` denotes such index in the
  `thresholds` array. The output index is computed as `X + offsets[t(X)]`.

  Args:
    file_infos: See `file_infos` argument of `index_shuffle` for details.

  Returns:
    A dictionary containing the following keys:
      - `files`, the vector of pathnames of files to read
      - `num_elements`, an integer identifying the total number of elements
      - `offsets`, the vector of offsets to use for index adjustment (in case
        any elements should be skipped)
      - `thresholds`, the vector of thresholds to use for index adjustment (in
        case any elements should be skipped)
  """
  files = []
  num_elements = 0
  offsets = np.int64([])
  offset_sum = 0
  thresholds = np.int64([])
  threshold_sum = 0
  adjustment_needed = False
  for file_info in file_infos:
    files.append(file_info["path"])
    skip = 0
    if "skip" in file_info:
      if file_info["skip"] < -1:
        raise ValueError("`skip` should be greater than `-1` but got {}".format(
            file_info["skip"]))
      if file_info["skip"] == -1:
        skip = file_info["num_elements"]
      else:
        skip = min(file_info["skip"], file_info["num_elements"])
    take = file_info["num_elements"] - skip
    if "take" in file_info:
      if file_info["take"] < -1:
        raise ValueError("`take` should be greater than `-1` but got {}".format(
            file_info["take"]))
      # `file_info["take"] == -1` is a no-op
      if file_info["take"] != -1:
        take = min(file_info["take"], take)
    remainder = file_info["num_elements"] - skip - take
    if take != file_info["num_elements"]:
      adjustment_needed = True
    num_elements += take
    offsets = np.append(offsets, offset_sum + skip)
    offset_sum += skip + remainder
    thresholds = np.append(thresholds, threshold_sum)
    threshold_sum += take
  result = {"files": files, "num_elements": num_elements}
  if adjustment_needed:
    result["offsets"] = offsets
    result["thresholds"] = thresholds
  return result


def _adjust_index(index, thresholds, offsets):
  """Adjusts index to account for elements to be skipped."""
  t_index = array_ops.shape(
      array_ops.boolean_mask(
          thresholds,
          math_ops.less_equal(thresholds, index)))[0] - 1
  return index + array_ops.gather(offsets, t_index)


# TODO(jsimsa): Expose this method in the public API. When we do, consider
# defining `FileInfo` as a public API to encapsulate the information provided
# through the `file_infos` argument.
def index_shuffle(file_infos,
                  reader_factory,
                  seed=None,
                  reshuffle_each_iteration=False,
                  num_parallel_calls=dataset_ops.AUTOTUNE):
  """Creates a (globally) shuffled dataset from the given set of files.

  Unlike `tf.data.Dataset.shuffle()`, which uses an in-memory buffer to shuffle
  elements of input dataset in a streaming fashion,
  `tf.data.experimental.index_shuffle()` performs a global shuffle of element
  indices and then reads the data in a shuffled order. The advantage of
  `index_shuffle()` is that it can perform global shuffle of datasets that do
  not fit into memory (as long as the array of their indices does) and that the
  shuffling logic it provides is compatible with symbolic checkpointing. The
  disadvantage of `index_shuffle()` is that reading data in a shuffled random
  order will in general not be as efficient as reading data sequentially.

  Args:
    file_infos: A list of dictionaries that describe each file of the input
      dataset. Each dictionary is expected to contain the "path" key, which
      identifies the path of the file and the "num_elements" key, which
      identifies the number of elements in the file. In addition, the "skip"
      and "take" keys can be used to identify the number of elements to skip
      and take respectively. By default, no elements are skipped and all
      elements are taken.
    reader_factory: A function that maps a sequence of filenames to an instance
      of `tf.data.Dataset` that reads data from the files.
    seed: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the random
      seed that will be used to shuffle the order of elements. Default to
      non-deterministic seed.
    reshuffle_each_iteration: (Optional.) A `tf.bool` scalar `tf.Tensor`, that
      determines whether to change the shuffle order each iteration. Defaults to
      `False`.
    num_parallel_calls: (Optional.) A `tf.int64` scalar `tf.Tensor`, that
      determines the maximum number of random access operations to perform
      in parallel. By default, the tf.data runtime uses autotuning to determine
      the value dynamically.

  Returns:
    A `tf.data.Dataset` object, representing a globally shuffled dataset of
    the input data.
  """

  result = _process_file_infos(file_infos)

  def sequential_index_shuffle(seeds):
    dataset = dataset_ops.Dataset.range(result["num_elements"])

    def read_element(dataset, index):
      # 1) Shuffle the index.
      shuffled_index = stateless_random_ops.index_shuffle(
          index, seeds, result["num_elements"] - 1)
      # 2) If needed, adjust the index to the non-contiguous range.
      if "thresholds" in result and "offsets" in result:
        shuffled_index = _adjust_index(shuffled_index, result["thresholds"],
                                       result["offsets"])
      # 3) Perform the read.
      return random_access.at(dataset, shuffled_index)

    # We evaluate `reader_factory()` eagerly to prevent the dataset from being
    # created on every lookup.
    map_func = functools.partial(read_element, reader_factory(result["files"]))
    return dataset.map(map_func, num_parallel_calls=num_parallel_calls)

  rng_ds = dataset_ops.Dataset.random(
      seed=seed,
      rerandomize_each_iteration=reshuffle_each_iteration)
  rng_ds = rng_ds.take(2).batch(2, drop_remainder=True)
  return rng_ds.flat_map(sequential_index_shuffle)
