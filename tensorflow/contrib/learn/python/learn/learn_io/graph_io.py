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

from tensorflow.contrib.input_pipeline.python.ops import input_pipeline_ops
from tensorflow.python import summary
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import variables as var_ops
from tensorflow.python.platform import gfile
from tensorflow.python.training import input as input_ops
from tensorflow.python.training import queue_runner


# Default name for key in the feature dict.
KEY_FEATURE_NAME = '__key__'


def read_batch_examples(file_pattern, batch_size, reader,
                        randomize_input=True, num_epochs=None,
                        queue_capacity=10000, num_threads=1,
                        read_batch_size=1, parse_fn=None,
                        name=None):
  """Adds operations to read, queue, batch `Example` protos.

  Given file pattern (or list of files), will setup a queue for file names,
  read `Example` proto using provided `reader`, use batch queue to create
  batches of examples of size `batch_size`.

  All queue runners are added to the queue runners collection, and may be
  started via `start_queue_runners`.

  All ops are added to the default graph.

  Use `parse_fn` if you need to do parsing / processing on single examples.

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
      `tf.global_variables_initializer()` as shown in the tests.
    queue_capacity: Capacity for input queue.
    num_threads: The number of threads enqueuing examples.
    read_batch_size: An int or scalar `Tensor` specifying the number of
      records to read at once
    parse_fn: Parsing function, takes `Example` Tensor returns parsed
      representation. If `None`, no parsing is done.
    name: Name of resulting op.

  Returns:
    String `Tensor` of batched `Example` proto.

  Raises:
    ValueError: for invalid inputs.
  """
  _, examples = read_keyed_batch_examples(
      file_pattern=file_pattern, batch_size=batch_size, reader=reader,
      randomize_input=randomize_input, num_epochs=num_epochs,
      queue_capacity=queue_capacity, num_threads=num_threads,
      read_batch_size=read_batch_size, parse_fn=parse_fn, name=name)
  return examples


def read_keyed_batch_examples(
    file_pattern, batch_size, reader,
    randomize_input=True, num_epochs=None,
    queue_capacity=10000, num_threads=1,
    read_batch_size=1, parse_fn=None,
    name=None):
  """Adds operations to read, queue, batch `Example` protos.

  Given file pattern (or list of files), will setup a queue for file names,
  read `Example` proto using provided `reader`, use batch queue to create
  batches of examples of size `batch_size`.

  All queue runners are added to the queue runners collection, and may be
  started via `start_queue_runners`.

  All ops are added to the default graph.

  Use `parse_fn` if you need to do parsing / processing on single examples.

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
      `tf.global_variables_initializer()` as shown in the tests.
    queue_capacity: Capacity for input queue.
    num_threads: The number of threads enqueuing examples.
    read_batch_size: An int or scalar `Tensor` specifying the number of
      records to read at once
    parse_fn: Parsing function, takes `Example` Tensor returns parsed
      representation. If `None`, no parsing is done.
    name: Name of resulting op.

  Returns:
    Returns tuple of:
    - `Tensor` of string keys.
    - String `Tensor` of batched `Example` proto.

  Raises:
    ValueError: for invalid inputs.
  """
  return _read_keyed_batch_examples_helper(
      file_pattern,
      batch_size,
      reader,
      randomize_input,
      num_epochs,
      queue_capacity,
      num_threads,
      read_batch_size,
      parse_fn,
      setup_shared_queue=False,
      name=name)


def _read_keyed_batch_examples_shared_queue(file_pattern,
                                            batch_size,
                                            reader,
                                            randomize_input=True,
                                            num_epochs=None,
                                            queue_capacity=10000,
                                            num_threads=1,
                                            read_batch_size=1,
                                            parse_fn=None,
                                            name=None):
  """Adds operations to read, queue, batch `Example` protos.

  Given file pattern (or list of files), will setup a shared queue for file
  names, setup a worker queue that pulls from the shared queue, read `Example`
  protos using provided `reader`, use batch queue to create batches of examples
  of size `batch_size`. This provides at most once visit guarantees. Note that
  this only works if the parameter servers are not pre-empted or restarted or
  the session is not restored from a checkpoint since the state of a queue
  is not checkpointed and we will end up restarting from the entire list of
  files.

  All queue runners are added to the queue runners collection, and may be
  started via `start_queue_runners`.

  All ops are added to the default graph.

  Use `parse_fn` if you need to do parsing / processing on single examples.

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
      `tf.global_variables_initializer()` as shown in the tests.
    queue_capacity: Capacity for input queue.
    num_threads: The number of threads enqueuing examples.
    read_batch_size: An int or scalar `Tensor` specifying the number of
      records to read at once
    parse_fn: Parsing function, takes `Example` Tensor returns parsed
      representation. If `None`, no parsing is done.
    name: Name of resulting op.

  Returns:
    Returns tuple of:
    - `Tensor` of string keys.
    - String `Tensor` of batched `Example` proto.

  Raises:
    ValueError: for invalid inputs.
  """
  return _read_keyed_batch_examples_helper(
      file_pattern,
      batch_size,
      reader,
      randomize_input,
      num_epochs,
      queue_capacity,
      num_threads,
      read_batch_size,
      parse_fn,
      setup_shared_queue=True,
      name=name)


def _get_file_names(file_pattern, randomize_input):
  """Parse list of file names from pattern, optionally shuffled.

  Args:
    file_pattern: File glob pattern, or list of strings.
    randomize_input: Whether to shuffle the order of file names.

  Returns:
    List of file names matching `file_pattern`.

  Raises:
    ValueError: If `file_pattern` is empty, or pattern matches no files.
  """
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
  return file_names


def _get_examples(file_name_queue, reader, num_threads, read_batch_size,
                  parse_fn):
  with ops.name_scope('read'):
    example_list = []
    for _ in range(num_threads):
      if read_batch_size > 1:
        keys, examples_proto = reader().read_up_to(file_name_queue,
                                                   read_batch_size)
      else:
        keys, examples_proto = reader().read(file_name_queue)
      if parse_fn:
        parsed_examples = parse_fn(examples_proto)
        # Map keys into example map because batch_join doesn't support
        # tuple of Tensor + dict.
        if isinstance(parsed_examples, dict):
          parsed_examples[KEY_FEATURE_NAME] = keys
          example_list.append(parsed_examples)
        else:
          example_list.append((keys, parsed_examples))
      else:
        example_list.append((keys, examples_proto))
    return example_list


def _read_keyed_batch_examples_helper(file_pattern,
                                      batch_size,
                                      reader,
                                      randomize_input=True,
                                      num_epochs=None,
                                      queue_capacity=10000,
                                      num_threads=1,
                                      read_batch_size=1,
                                      parse_fn=None,
                                      setup_shared_queue=False,
                                      name=None):
  """Adds operations to read, queue, batch `Example` protos.

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
      `tf.global_variables_initializer()` as shown in the tests.
    queue_capacity: Capacity for input queue.
    num_threads: The number of threads enqueuing examples.
    read_batch_size: An int or scalar `Tensor` specifying the number of
      records to read at once
    parse_fn: Parsing function, takes `Example` Tensor returns parsed
      representation. If `None`, no parsing is done.
    setup_shared_queue: Whether to set up a shared queue for file names.
    name: Name of resulting op.

  Returns:
    Returns tuple of:
    - `Tensor` of string keys.
    - String `Tensor` of batched `Example` proto.

  Raises:
    ValueError: for invalid inputs.
  """
  # Retrieve files to read.
  file_names = _get_file_names(file_pattern, randomize_input)

  # Check input parameters are given and reasonable.
  if (not queue_capacity) or (queue_capacity <= 0):
    raise ValueError('Invalid queue_capacity %s.' % queue_capacity)
  if (batch_size is None) or (
      (not isinstance(batch_size, ops.Tensor)) and
      (batch_size <= 0 or batch_size > queue_capacity)):
    raise ValueError(
        'Invalid batch_size %s, with queue_capacity %s.' %
        (batch_size, queue_capacity))
  if (read_batch_size is None) or (
      (not isinstance(read_batch_size, ops.Tensor)) and
      (read_batch_size <= 0)):
    raise ValueError('Invalid read_batch_size %s.' % read_batch_size)
  if (not num_threads) or (num_threads <= 0):
    raise ValueError('Invalid num_threads %s.' % num_threads)
  if (num_epochs is not None) and (num_epochs <= 0):
    raise ValueError('Invalid num_epochs %s.' % num_epochs)

  with ops.name_scope(name, 'read_batch_examples', [file_pattern]) as scope:
    with ops.name_scope('file_name_queue') as file_name_queue_scope:
      if setup_shared_queue:
        file_name_queue = data_flow_ops.FIFOQueue(
            capacity=1, dtypes=[dtypes.string], shapes=[[]])
        enqueue_op = file_name_queue.enqueue(
            input_pipeline_ops.seek_next(
                file_names, shuffle=randomize_input, num_epochs=num_epochs))
        queue_runner.add_queue_runner(
            queue_runner.QueueRunner(file_name_queue, [enqueue_op]))
      else:
        file_name_queue = input_ops.string_input_producer(
            constant_op.constant(
                file_names, name='input'),
            shuffle=randomize_input,
            num_epochs=num_epochs,
            name=file_name_queue_scope)

    example_list = _get_examples(file_name_queue, reader, num_threads,
                                 read_batch_size, parse_fn)

    enqueue_many = read_batch_size > 1

    if num_epochs is None:
      allow_smaller_final_batch = False
    else:
      allow_smaller_final_batch = True

    # Setup batching queue given list of read example tensors.
    if randomize_input:
      if isinstance(batch_size, ops.Tensor):
        min_after_dequeue = int(queue_capacity * 0.4)
      else:
        min_after_dequeue = max(queue_capacity - (3 * batch_size), batch_size)
      queued_examples_with_keys = input_ops.shuffle_batch_join(
          example_list, batch_size, capacity=queue_capacity,
          min_after_dequeue=min_after_dequeue,
          enqueue_many=enqueue_many, name=scope,
          allow_smaller_final_batch=allow_smaller_final_batch)
    else:
      queued_examples_with_keys = input_ops.batch_join(
          example_list, batch_size, capacity=queue_capacity,
          enqueue_many=enqueue_many, name=scope,
          allow_smaller_final_batch=allow_smaller_final_batch)
    if parse_fn and isinstance(queued_examples_with_keys, dict):
      queued_keys = queued_examples_with_keys.pop(KEY_FEATURE_NAME)
      return queued_keys, queued_examples_with_keys
    return queued_examples_with_keys


def read_keyed_batch_features(file_pattern,
                              batch_size,
                              features,
                              reader,
                              randomize_input=True,
                              num_epochs=None,
                              queue_capacity=10000,
                              reader_num_threads=1,
                              feature_queue_capacity=100,
                              num_enqueue_threads=2,
                              parse_fn=None,
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
      tf.local_variables_initializer() as shown in the tests.
    queue_capacity: Capacity for input queue.
    reader_num_threads: The number of threads to read examples.
    feature_queue_capacity: Capacity of the parsed features queue.
    num_enqueue_threads: Number of threads to enqueue the parsed example queue.
      Using multiple threads to enqueue the parsed example queue helps maintain
      a full queue when the subsequent computations overall are cheaper than
      parsing.
    parse_fn: Parsing function, takes `Example` Tensor returns parsed
      representation. If `None`, no parsing is done.
    name: Name of resulting op.

  Returns:
    Returns tuple of:
    - `Tensor` of string keys.
    - A dict of `Tensor` or `SparseTensor` objects for each in `features`.

  Raises:
    ValueError: for invalid inputs.
  """

  with ops.name_scope(name, 'read_batch_features', [file_pattern]) as scope:
    keys, examples = read_keyed_batch_examples(
        file_pattern, batch_size, reader, randomize_input=randomize_input,
        num_epochs=num_epochs, queue_capacity=queue_capacity,
        num_threads=reader_num_threads, read_batch_size=batch_size,
        parse_fn=parse_fn, name=scope)
    # Parse the example.
    feature_map = parsing_ops.parse_example(examples, features)
    return queue_parsed_features(
        feature_map,
        keys=keys,
        feature_queue_capacity=feature_queue_capacity,
        num_enqueue_threads=num_enqueue_threads,
        name=scope)


def _read_keyed_batch_features_shared_queue(file_pattern,
                                            batch_size,
                                            features,
                                            reader,
                                            randomize_input=True,
                                            num_epochs=None,
                                            queue_capacity=10000,
                                            reader_num_threads=1,
                                            feature_queue_capacity=100,
                                            num_queue_runners=2,
                                            parse_fn=None,
                                            name=None):
  """Adds operations to read, queue, batch and parse `Example` protos.

  Given file pattern (or list of files), will setup a shared queue for file
  names, setup a worker queue that gets filenames from the shared queue,
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
      tf.local_variables_initializer() as shown in the tests.
    queue_capacity: Capacity for input queue.
    reader_num_threads: The number of threads to read examples.
    feature_queue_capacity: Capacity of the parsed features queue.
    num_queue_runners: Number of threads to enqueue the parsed example queue.
      Using multiple threads to enqueue the parsed example queue helps maintain
      a full queue when the subsequent computations overall are cheaper than
      parsing.
    parse_fn: Parsing function, takes `Example` Tensor returns parsed
      representation. If `None`, no parsing is done.
    name: Name of resulting op.

  Returns:
    Returns tuple of:
    - `Tensor` of string keys.
    - A dict of `Tensor` or `SparseTensor` objects for each in `features`.

  Raises:
    ValueError: for invalid inputs.
  """

  with ops.name_scope(name, 'read_batch_features', [file_pattern]) as scope:
    keys, examples = _read_keyed_batch_examples_shared_queue(
        file_pattern,
        batch_size,
        reader,
        randomize_input=randomize_input,
        num_epochs=num_epochs,
        queue_capacity=queue_capacity,
        num_threads=reader_num_threads,
        read_batch_size=batch_size,
        parse_fn=parse_fn,
        name=scope)
    # Parse the example.
    feature_map = parsing_ops.parse_example(examples, features)
    return queue_parsed_features(
        feature_map,
        keys=keys,
        feature_queue_capacity=feature_queue_capacity,
        num_enqueue_threads=num_queue_runners,
        name=scope)


def queue_parsed_features(parsed_features,
                          keys=None,
                          feature_queue_capacity=100,
                          num_enqueue_threads=2,
                          name=None):
  """Speeds up parsing by using queues to do it asynchronously.

  This function adds the tensors in `parsed_features` to a queue, which allows
  the parsing (or any other expensive op before this) to be asynchronous wrt the
  rest of the training graph. This greatly improves read latency and speeds up
  training since the data will already be parsed and ready when each step of
  training needs it.

  All queue runners are added to the queue runners collection, and may be
  started via `start_queue_runners`.

  All ops are added to the default graph.

  Args:
    parsed_features: A dict of string key to `Tensor` or `SparseTensor` objects.
    keys: `Tensor` of string keys.
    feature_queue_capacity: Capacity of the parsed features queue.
    num_enqueue_threads: Number of threads to enqueue the parsed example queue.
      Using multiple threads to enqueue the parsed example queue helps maintain
      a full queue when the subsequent computations overall are cheaper than
      parsing.
    name: Name of resulting op.

  Returns:
    Returns tuple of:
    - `Tensor` corresponding to `keys` if provided, otherwise `None`.
    -  A dict of string key to `Tensor` or `SparseTensor` objects corresponding
       to `parsed_features`.
  Raises:
    ValueError: for invalid inputs.
  """

  args = list(parsed_features.values())
  if keys is not None:
    args += [keys]

  with ops.name_scope(name, 'queue_parsed_features', args):
    # Lets also add preprocessed tensors into the queue types for each item of
    # the queue.
    tensors_to_enqueue = []
    # Each entry contains the key, and a boolean which indicates whether the
    # tensor was a sparse tensor.
    tensors_mapping = []
    # TODO(sibyl-Aix6ihai): Most of the functionality here is about pushing sparse
    # tensors into a queue. This could be taken care in somewhere else so others
    # can reuse it. Also, QueueBase maybe extended to handle sparse tensors
    # directly.
    for key in sorted(parsed_features.keys()):
      tensor = parsed_features[key]
      if isinstance(tensor, sparse_tensor.SparseTensor):
        tensors_mapping.append((key, True))
        tensors_to_enqueue.extend([
            tensor.indices, tensor.values, tensor.dense_shape])
      else:
        tensors_mapping.append((key, False))
        tensors_to_enqueue.append(tensor)

    if keys is not None:
      tensors_to_enqueue.append(keys)

    queue_dtypes = [x.dtype for x in tensors_to_enqueue]
    input_queue = data_flow_ops.FIFOQueue(feature_queue_capacity, queue_dtypes)

    # Add a summary op to debug if our feature queue is full or not.
    summary.scalar('queue/parsed_features/%s/fraction_of_%d_full' %
                   (input_queue.name, feature_queue_capacity),
                   math_ops.cast(input_queue.size(), dtypes.float32) *
                   (1. / feature_queue_capacity))

    # Use a single QueueRunner with multiple threads to enqueue so the queue is
    # always full. The threads are coordinated so the last batch will not be
    # lost.
    enqueue_ops = [input_queue.enqueue(tensors_to_enqueue)
                   for _ in range(num_enqueue_threads)]
    queue_runner.add_queue_runner(queue_runner.QueueRunner(
        input_queue, enqueue_ops,
        queue_closed_exception_types=(errors.OutOfRangeError,
                                      errors.CancelledError)))

    dequeued_tensors = input_queue.dequeue()

    # Reset shapes on dequeued tensors.
    for i in range(len(tensors_to_enqueue)):
      dequeued_tensors[i].set_shape(tensors_to_enqueue[i].get_shape())

    # Recreate feature mapping according to the original dictionary.
    dequeued_parsed_features = {}
    index = 0
    for key, is_sparse_tensor in tensors_mapping:
      if is_sparse_tensor:
        # Three tensors are (indices, values, shape).
        dequeued_parsed_features[key] = sparse_tensor.SparseTensor(
            dequeued_tensors[index], dequeued_tensors[index + 1],
            dequeued_tensors[index + 2])
        index += 3
      else:
        dequeued_parsed_features[key] = dequeued_tensors[index]
        index += 1

    dequeued_keys = None
    if keys is not None:
      dequeued_keys = dequeued_tensors[-1]

    return dequeued_keys, dequeued_parsed_features


def read_batch_features(file_pattern,
                        batch_size,
                        features,
                        reader,
                        randomize_input=True,
                        num_epochs=None,
                        queue_capacity=10000,
                        feature_queue_capacity=100,
                        reader_num_threads=1,
                        parse_fn=None,
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
      tf.local_variables_initializer() as shown in the tests.
    queue_capacity: Capacity for input queue.
    feature_queue_capacity: Capacity of the parsed features queue. Set this
      value to a small number, for example 5 if the parsed features are large.
    reader_num_threads: The number of threads to read examples.
    parse_fn: Parsing function, takes `Example` Tensor returns parsed
      representation. If `None`, no parsing is done.
    name: Name of resulting op.

  Returns:
    A dict of `Tensor` or `SparseTensor` objects for each in `features`.

  Raises:
    ValueError: for invalid inputs.
  """
  _, features = read_keyed_batch_features(
      file_pattern, batch_size, features, reader,
      randomize_input=randomize_input, num_epochs=num_epochs,
      queue_capacity=queue_capacity,
      feature_queue_capacity=feature_queue_capacity,
      reader_num_threads=reader_num_threads,
      parse_fn=parse_fn, name=name)
  return features


def read_batch_record_features(file_pattern, batch_size, features,
                               randomize_input=True, num_epochs=None,
                               queue_capacity=10000, reader_num_threads=1,
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
      tf.local_variables_initializer() as shown in the tests.
    queue_capacity: Capacity for input queue.
    reader_num_threads: The number of threads to read examples.
    name: Name of resulting op.

  Returns:
    A dict of `Tensor` or `SparseTensor` objects for each in `features`.

  Raises:
    ValueError: for invalid inputs.
  """
  return read_batch_features(
      file_pattern=file_pattern,
      batch_size=batch_size,
      features=features,
      reader=io_ops.TFRecordReader,
      randomize_input=randomize_input,
      num_epochs=num_epochs,
      queue_capacity=queue_capacity,
      reader_num_threads=reader_num_threads,
      name=name)
