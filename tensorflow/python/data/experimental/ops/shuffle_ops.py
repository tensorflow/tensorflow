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

# The following constant determines the (maximum) number of file groups to use
# for index-based shuffling. A dataset reader will be constructed for each file
# group and the number of file groups is kept constant to avoid the size of the
# dataset graph to be linear in the number of files (which could slow down graph
# construction). This constant also determines the parallelism of the graph
# execution and the current value has been empirically chosen to offer a good
# tradeoff between performance of graph construction and graph execution.
_NUM_INDEX_SHUFFLE_FILE_GROUPS = 10


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


# TODO(jsimsa): Expose this method in the public API. When you do, define
# `FileInfo` class to encapsulate the information provided through the
# `file_infos` argument.
def index_shuffle(file_infos,
                  reader_factory,
                  seed=None,
                  reshuffle_each_iteration=False):
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

  Returns:
    A `tf.data.Dataset` object, representing globbally shuffled dataset of
    the input data.
  """

  def idx(i):
    return i % _NUM_INDEX_SHUFFLE_FILE_GROUPS

  # Create a sequence of (file_index, record_index) pairs.
  #
  # TODO(jsimsa): Implement this using tf.data and TF ops.
  indices = []
  counters = [0] * _NUM_INDEX_SHUFFLE_FILE_GROUPS
  file_groups = [[] for _ in range(_NUM_INDEX_SHUFFLE_FILE_GROUPS)]
  for i, file_info in enumerate(file_infos):
    num_elements = file_info["num_elements"]
    skip = 0
    if "skip" in file_info:
      skip = file_info["skip"]
      if skip == -1:
        num_elements = 0
      else:
        num_elements = max(0, num_elements - skip)
    if "take" in file_info and file_info["take"] >= 0:
      num_elements = min(num_elements, file_info["take"])
    indices.extend([
        (idx(i), j + skip + counters[idx(i)]) for j in range(num_elements)
    ])
    counters[idx(i)] += file_info["num_elements"]
    file_groups[idx(i)].append(file_info["path"])

  if not indices:
    return dataset_ops.Dataset.from_tensor_slices([])

  def make_shuffled_dataset(seeds):
    # First component of the shuffled sequence of `(file_index, record_index)`
    # pairs identifies the order in which elements should be read from
    # different files.
    shuffled_indices = stateless_random_ops.stateless_shuffle(
        np.int64(indices), seeds)
    file_indices = shuffled_indices[:, 0]
    choice_dataset = dataset_ops.Dataset.from_tensor_slices(file_indices)
    datasets = []
    # Second component of the shuffled sequence of `(file_,id, record_index)`
    # pairs identifies the records to read (relative to a file).
    record_indices = shuffled_indices[:, 1]
    for i, file_group in enumerate(file_groups):
      if not file_group:
        break

      # Create a dataset that contains the order in which records should be
      # read from this file, using `file_indices` to select subsequence of
      # `record_indices` for the i-th file.
      dataset = dataset_ops.Dataset.from_tensor_slices(
          array_ops.boolean_mask(record_indices,
                                 math_ops.equal(file_indices, np.int64(i))))

      def read_element(dataset, index):
        return random_access.at(dataset, index)

      # Evaluate `reader_factory()` eagerly to avoid the dataset being created
      # on every lookup.
      map_func = functools.partial(read_element, reader_factory(file_group))

      # NOTE: The number of product of file groups and parallel reads should
      # not be much greater than the size of the threadpool (which defaults to
      # the number of available CPUs).
      dataset = dataset.map(map_func, num_parallel_calls=4)
      datasets.append(dataset.prefetch(1))
    return dataset_ops.Dataset.choose_from_datasets(
        datasets, choice_dataset, stop_on_empty_dataset=False)

  rng_ds = dataset_ops.Dataset.random(
      seed=seed,
      rerandomize_each_iteration=reshuffle_each_iteration).take(2).batch(2)
  return rng_ds.flat_map(make_shuffled_dataset)
