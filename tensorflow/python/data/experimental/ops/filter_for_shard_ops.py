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
"""Naive shard dataset transformation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import tf_export


@tf_export("data.experimental.filter_for_shard")
def filter_for_shard(num_shards, shard_index):
  """Creates a `Dataset` that includes only 1/`num_shards` of this dataset.

  This dataset operator is very useful when running distributed training, as
  it allows each worker to read a unique subset.

  When reading a single input file, you can skip elements as follows:

  ```python
  d = tf.data.TFRecordDataset(FLAGS.input_file)
  d = d.apply(tf.data.experimental.naive_shard(FLAGS.num_workers,
                                               FLAGS.worker_index))
  d = d.repeat(FLAGS.num_epochs)
  d = d.shuffle(FLAGS.shuffle_buffer_size)
  d = d.map(parser_fn, num_parallel_calls=FLAGS.num_map_threads)
  ```

  Important caveats:

  - Be sure to shard before you use any randomizing operator (such as
    shuffle).
  - Generally it is best if the shard operator is used early in the dataset
    pipeline. For example, when reading from a set of TFRecord files, shard
    before converting the dataset to input samples. This avoids reading every
    file on every worker. The following is an example of an efficient
    sharding strategy within a complete pipeline:

  ```python
  d = Dataset.list_files(FLAGS.pattern)
  d = d.apply(tf.data.experimental.naive_shard(FLAGS.num_workers,
                                               FLAGS.worker_index))
  d = d.repeat(FLAGS.num_epochs)
  d = d.shuffle(FLAGS.shuffle_buffer_size)
  d = d.interleave(tf.data.TFRecordDataset,
                   cycle_length=FLAGS.num_readers, block_length=1)
  d = d.map(parser_fn, num_parallel_calls=FLAGS.num_map_threads)
  ```

  Args:
    num_shards: A `tf.int64` scalar `tf.Tensor`, representing the number of
      shards operating in parallel.
    shard_index: A `tf.int64` scalar `tf.Tensor`, representing the worker index.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.

  Raises:
    ValueError: if `num_shards` or `shard_index` are illegal values. Note: error
      checking is done on a best-effort basis, and errors aren't guaranteed to
      be caught upon dataset creation. (e.g. providing in a placeholder tensor
      bypasses the early checking, and will instead result in an error during
      a session.run call.)
  """
  num_shards = ops.convert_to_tensor(
      num_shards, name="num_shards", dtype=dtypes.int64)
  num_shards_static = tensor_util.constant_value(num_shards)
  shard_index = ops.convert_to_tensor(shard_index, name="shard_index",
                                      dtype=dtypes.int64)
  shard_index_static = tensor_util.constant_value(shard_index)

  if num_shards_static is not None and num_shards_static < 1:
    raise ValueError("num_shards must be >= 1; got: %s" % num_shards_static)
  if shard_index_static is not None and shard_index_static < 0:
    raise ValueError("shard_index must be >= 0; got: %s" % shard_index_static)
  if (shard_index_static is not None and num_shards_static is not None and
      shard_index_static >= num_shards_static):
    raise ValueError("shard_index must be < num_shards; %s is not < %s" %
                     (shard_index_static, num_shards_static))

  def filter_fn(elem_index, _):
    mod_result = math_ops.mod(elem_index, num_shards)
    return math_ops.equal(mod_result, shard_index)

  def _apply_fn(dataset):
    # pylint: disable=protected-access
    return dataset._enumerate().filter(filter_fn).map(lambda _, elem: elem)

  return _apply_fn
