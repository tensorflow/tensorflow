"""Methods to read data in the graph."""
#  Copyright 2016 Google Inc. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import gfile
from tensorflow.python.training import input as input_ops


def read_batch_examples(file_pattern, batch_size, reader,
                        randomize_input=True, queue_capacity=10000,
                        num_threads=1, name='dequeue_examples'):
  """Adds operations to read, queue, batch `Example` protos.

  Given file pattern (or list of files), will setup a queue for file names,
  read `Example` proto using provided `reader`, use batch queue to create
  batches of examples of size `batch_size`.

  All queue runners are added to the queue runners collection, and may be
  started via `start_queue_runners`.

  All ops are added to the default graph.

  Args:
    file_pattern: List of files or pattern of file paths containing
        `Example` records. See `tf.gfile.Glob` for pattern rules.
    batch_size: An int or scalar `Tensor` specifying the batch size to use.
    reader: A function or class that returns an object with
      `read` method, (filename tensor) -> (example tensor).
    randomize_input: Whether the input should be randomized.
    queue_capacity: Capacity for input queue.
    num_threads: The number of threads enqueuing examples.
    name: Name of resulting op.

  Returns:
    String `Tensor` of batched `Example` proto.

  Raises:
    ValueError: for invalid inputs.
  """
  # Retrive files to read.
  if isinstance(file_pattern, list):
    file_names = file_pattern
    if not file_names:
      raise ValueError('No files given to dequeue_examples.')
  else:
    file_names = list(gfile.Glob(file_pattern))
    if not file_names:
      raise ValueError('No files match %s.' % file_pattern)

  # Sort files so it will be deterministic for unit tests. They'll be shuffled
  # in `string_input_producer` if `randomize_input` is enabled.
  if not randomize_input:
    file_names = sorted(file_names)

  # Check input parameters are given and reasonable.
  if (not queue_capacity) or (queue_capacity <= 0):
    raise ValueError('Invalid queue_capacity %s.' % queue_capacity)
  if (batch_size is None) or (
      (not isinstance(batch_size, ops.Tensor)) and
      (batch_size <= 0 or batch_size > queue_capacity)):
    raise ValueError(
        'Invalid batch_size %s, with queue_capacity %s.' %
        (batch_size, queue_capacity))
  if (not num_threads) or (num_threads <= 0):
    raise ValueError('Invalid num_threads %s.' % num_threads)

  with ops.name_scope(name) as scope:
    # Setup filename queue with shuffling.
    with ops.name_scope('file_name_queue') as file_name_queue_scope:
      file_name_queue = input_ops.string_input_producer(
          constant_op.constant(file_names, name='input'),
          shuffle=randomize_input, name=file_name_queue_scope)

    # Create reader and set it to read from filename queue.
    with ops.name_scope('read'):
      _, example_proto = reader().read(file_name_queue)

    # Setup batching queue.
    if randomize_input:
      if isinstance(batch_size, ops.Tensor):
        min_after_dequeue = int(queue_capacity * 0.4)
      else:
        min_after_dequeue = max(queue_capacity - (3 * batch_size), batch_size)
      examples = input_ops.shuffle_batch(
          [example_proto], batch_size, capacity=queue_capacity,
          num_threads=num_threads, min_after_dequeue=min_after_dequeue,
          name=scope)
    else:
      examples = input_ops.batch(
          [example_proto], batch_size, capacity=queue_capacity,
          num_threads=num_threads, name=scope)

    return examples


def read_batch_features(file_pattern, batch_size, features, reader,
                        randomize_input=True, queue_capacity=10000,
                        num_threads=1, name='dequeue_examples'):
  """Adds operations to read, queue, batch and parse `Example` protos.

  Given file pattern (or list of files), will setup a queue for file names,
  read `Example` proto using provided `reader`, use batch queue to create
  batches of examples of size `batch_size` and parse example given `features`
  specification.

  All queue runners are added to the queue runners collection, and may be
  started via `start_queue_runners`.

  All ops are added to the default graph.

  Args:
    file_pattern: List of files or pattern of file paths containing
        `Example` records. See `tf.gfile.Glob` for pattern rules.
    batch_size: An int or scalar `Tensor` specifying the batch size to use.
    features: A `dict` mapping feature keys to `FixedLenFeature` or
      `VarLenFeature` values.
    reader: A function or class that returns an object with
      `read` method, (filename tensor) -> (example tensor).
    randomize_input: Whether the input should be randomized.
    queue_capacity: Capacity for input queue.
    num_threads: The number of threads enqueuing examples.
    name: Name of resulting op.

  Returns:
    A dict of `Tensor` or `SparseTensor` objects for each in `features`.

  Raises:
    ValueError: for invalid inputs.
  """
  examples = read_batch_examples(
      file_pattern, batch_size, reader, randomize_input,
      queue_capacity, num_threads, name=name)

  # Parse features into tensors.
  return parsing_ops.parse_example(examples, features)


def read_batch_record_features(file_pattern, batch_size, features,
                               randomize_input=True, queue_capacity=10000,
                               num_threads=1, name='dequeue_record_examples'):
  """Reads TFRecord, queues, batches and parses `Example` proto.

  See more detailed description in `read_examples`.

  Args:
    file_pattern: List of files or pattern of file paths containing
        `Example` records. See `tf.gfile.Glob` for pattern rules.
    batch_size: An int or scalar `Tensor` specifying the batch size to use.
    features: A `dict` mapping feature keys to `FixedLenFeature` or
      `VarLenFeature` values.
    randomize_input: Whether the input should be randomized.
    queue_capacity: Capacity for input queue.
    num_threads: The number of threads enqueuing examples.
    name: Name of resulting op.

  Returns:
    A dict of `Tensor` or `SparseTensor` objects for each in `features`.

  Raises:
    ValueError: for invalid inputs.
  """
  return read_batch_features(
      file_pattern=file_pattern, batch_size=batch_size, features=features,
      reader=io_ops.TFRecordReader,
      randomize_input=randomize_input,
      queue_capacity=queue_capacity, num_threads=num_threads, name=name)
