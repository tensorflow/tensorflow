# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Methods to read data in the graph."""

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
                        randomize_input=True, num_epochs=None,
                        queue_capacity=10000, num_threads=1,
                        name=None):
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
    num_epochs: Integer specifying the number of times to read through the
      dataset. If `None`, cycles through the dataset forever.
      NOTE - If specified, creates a variable that must be initialized, so call
      `tf.initialize_all_variables()` as shown in the tests.
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
  if (num_epochs is not None) and (num_epochs <= 0):
    raise ValueError('Invalid num_epochs %s.' % num_epochs)

  with ops.op_scope([file_pattern], name, 'read_batch_examples') as scope:
    # Setup filename queue with shuffling.
    with ops.name_scope('file_name_queue') as file_name_queue_scope:
      file_name_queue = input_ops.string_input_producer(
          constant_op.constant(file_names, name='input'),
          shuffle=randomize_input, num_epochs=num_epochs,
          name=file_name_queue_scope)

    # Create readers, one per thread and set them to read from filename queue.
    with ops.name_scope('read'):
      example_list = []
      for _ in range(num_threads):
        _, example_proto = reader().read(file_name_queue)
        example_list.append([example_proto])

    # Setup batching queue given list of read example tensors.
    if randomize_input:
      if isinstance(batch_size, ops.Tensor):
        min_after_dequeue = int(queue_capacity * 0.4)
      else:
        min_after_dequeue = max(queue_capacity - (3 * batch_size), batch_size)
      examples = input_ops.shuffle_batch_join(
          example_list, batch_size, capacity=queue_capacity,
          min_after_dequeue=min_after_dequeue,
          name=scope)
    else:
      examples = input_ops.batch_join(
          example_list, batch_size, capacity=queue_capacity,
          name=scope)

    return examples


def read_batch_features(file_pattern, batch_size, features, reader,
                        randomize_input=True, num_epochs=None,
                        queue_capacity=10000, reader_num_threads=1,
                        parser_num_threads=1,
                        name=None):
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
    num_epochs: Integer specifying the number of times to read through the
      dataset. If None, cycles through the dataset forever. NOTE - If specified,
      creates a variable that must be initialized, so call
      tf.initialize_all_variables() as shown in the tests.
    queue_capacity: Capacity for input queue.
    reader_num_threads: The number of threads to read examples.
    parser_num_threads: The number of threads to parse examples.
    name: Name of resulting op.

  Returns:
    A dict of `Tensor` or `SparseTensor` objects for each in `features`.

  Raises:
    ValueError: for invalid inputs.
  """
  with ops.op_scope([file_pattern], name, 'read_batch_features') as scope:
    examples = read_batch_examples(
        file_pattern, batch_size, reader, randomize_input=randomize_input,
        num_epochs=num_epochs, queue_capacity=queue_capacity,
        num_threads=reader_num_threads, name=scope)

    if parser_num_threads == 1:
      # Avoid queue overhead for single thread
      return parsing_ops.parse_example(examples, features)
    else:
      # Parse features into tensors in many threads and put on the queue.
      features_list = []
      for _ in range(parser_num_threads):
        features_list.append(parsing_ops.parse_example(examples, features))
      return input_ops.batch_join(
          features_list,
          batch_size=batch_size,
          capacity=queue_capacity,
          enqueue_many=True,
          name='parse_example_batch_join')


def read_batch_record_features(file_pattern, batch_size, features,
                               randomize_input=True, num_epochs=None,
                               queue_capacity=10000, reader_num_threads=1,
                               parser_num_threads=1,
                               name='dequeue_record_examples'):
  """Reads TFRecord, queues, batches and parses `Example` proto.

  See more detailed description in `read_examples`.

  Args:
    file_pattern: List of files or pattern of file paths containing
        `Example` records. See `tf.gfile.Glob` for pattern rules.
    batch_size: An int or scalar `Tensor` specifying the batch size to use.
    features: A `dict` mapping feature keys to `FixedLenFeature` or
      `VarLenFeature` values.
    randomize_input: Whether the input should be randomized.
    num_epochs: Integer specifying the number of times to read through the
      dataset. If None, cycles through the dataset forever. NOTE - If specified,
      creates a variable that must be initialized, so call
      tf.initialize_all_variables() as shown in the tests.
    queue_capacity: Capacity for input queue.
    reader_num_threads: The number of threads to read examples.
    parser_num_threads: The number of threads to parse examples.
    name: Name of resulting op.

  Returns:
    A dict of `Tensor` or `SparseTensor` objects for each in `features`.

  Raises:
    ValueError: for invalid inputs.
  """
  return read_batch_features(
      file_pattern=file_pattern, batch_size=batch_size, features=features,
      reader=io_ops.TFRecordReader,
      randomize_input=randomize_input, num_epochs=num_epochs,
      queue_capacity=queue_capacity, reader_num_threads=reader_num_threads,
      parser_num_threads=parser_num_threads, name=name)
