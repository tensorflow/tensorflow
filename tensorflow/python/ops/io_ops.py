# Copyright 2015 Google Inc. All Rights Reserved.
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

# pylint: disable=line-too-long
"""## Placeholders

TensorFlow provides a placeholder operation that must be fed with data
on execution.  For more info, see the section on [Feeding
data](../../how_tos/reading_data/index.md#feeding).

@@placeholder
@@placeholder_with_default

For feeding `SparseTensor`s which are composite type,
there is a convenience function:

@@sparse_placeholder

## Readers

TensorFlow provides a set of Reader classes for reading data formats.
For more information on inputs and readers, see [Reading
data](../../how_tos/reading_data/index.md).

@@ReaderBase
@@TextLineReader
@@WholeFileReader
@@IdentityReader
@@TFRecordReader
@@FixedLengthRecordReader

## Converting

TensorFlow provides several operations that you can use to convert various data
formats into tensors.

@@decode_csv
@@decode_raw

- - -

### Example protocol buffer

TensorFlow's [recommended format for training
examples](../../how_tos/reading_data/index.md#standard-tensorflow-format)
is serialized `Example` protocol buffers, [described
here](https://www.tensorflow.org/code/tensorflow/core/example/example.proto).
They contain `Features`, [described
here](https://www.tensorflow.org/code/tensorflow/core/example/feature.proto).

@@VarLenFeature
@@FixedLenFeature
@@FixedLenSequenceFeature
@@parse_example
@@parse_single_example
@@decode_json_example

## Queues

TensorFlow provides several implementations of 'Queues', which are
structures within the TensorFlow computation graph to stage pipelines
of tensors together. The following describe the basic Queue interface
and some implementations.  To see an example use, see [Threading and
Queues](../../how_tos/threading_and_queues/index.md).

@@QueueBase
@@FIFOQueue
@@RandomShuffleQueue

## Dealing with the filesystem

@@matching_files
@@read_file

## Input pipeline

TensorFlow functions for setting up an input-prefetching pipeline.
Please see the [reading data how-to](../../how_tos/reading_data/index.md)
for context.

### Beginning of an input pipeline

The "producer" functions add a queue to the graph and a corresponding
`QueueRunner` for running the subgraph that fills that queue.

@@match_filenames_once
@@limit_epochs
@@input_producer
@@range_input_producer
@@slice_input_producer
@@string_input_producer

### Batching at the end of an input pipeline

These functions add a queue to the graph to assemble a batch of
examples, with possible shuffling.  They also add a `QueueRunner` for
running the subgraph that fills that queue.

Use [`batch`](#batch) or [`batch_join`](#batch_join) for batching
examples that have already been well shuffled.  Use
[`shuffle_batch`](#shuffle_batch) or
[`shuffle_batch_join`](#shuffle_batch_join) for examples that would
benefit from additional shuffling.

Use [`batch`](#batch) or [`shuffle_batch`](#shuffle_batch) if you want a
single thread producing examples to batch, or if you have a
single subgraph producing examples but you want to run it in *N* threads
(where you increase *N* until it can keep the queue full).  Use
[`batch_join`](#batch_join) or [`shuffle_batch_join`](#shuffle_batch_join)
if you have *N* different subgraphs producing examples to batch and you
want them run by *N* threads.

@@batch
@@batch_join
@@shuffle_batch
@@shuffle_batch_join
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import common_shapes
from tensorflow.python.ops import gen_io_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_io_ops import *
# pylint: enable=wildcard-import


# pylint: disable=protected-access
def _save(filename, tensor_names, tensors, tensor_slices=None, name="save"):
  """Save a list of tensors to a file with given names.

  Example usage without slice info:
    Save("/foo/bar", ["w", "b"], [w, b])

  Example usage with slices:
    Save("/foo/bar", ["w", "w"], [slice0, slice1],
         tensor_slices=["4 10 0,2:-", "4 10 2,2:-"])

  Args:
    filename: the file name of the sstable.
    tensor_names: a list of strings.
    tensors: the list of tensors to be saved.
    tensor_slices: Optional list of strings to specify the shape and slices of
      a larger virtual tensor that each tensor is a part of.  If not specified
      each tensor is saved as a full slice.
    name: string.  Optional name for the op.

  Requires:
    The length of tensors should match the size of tensor_names and of
    tensor_slices.

  Returns:
    An Operation that saves the tensors.
  """
  if tensor_slices is None:
    return gen_io_ops._save(filename, tensor_names, tensors, name=name)
  else:
    return gen_io_ops._save_slices(filename, tensor_names, tensor_slices,
                                   tensors, name=name)


def _restore_slice(file_pattern, tensor_name, shape_and_slice, tensor_type,
                   name="restore_slice", preferred_shard=-1):
  """Restore a tensor slice from a set of files with a given pattern.

  Example usage:
    RestoreSlice("/foo/bar-?????-of-?????", "w", "10 10 0,2:-", DT_FLOAT)

  Args:
    file_pattern: the file pattern used to match a set of checkpoint files.
    tensor_name: the name of the tensor to restore.
    shape_and_slice: the shape-and-slice spec of the slice.
    tensor_type: the type of the tensor to restore.
    name: string.  Optional name for the op.
    preferred_shard: Int. Optional shard to open first in the checkpoint file.

  Returns:
    A tensor of type "tensor_type".
  """
  base_type = dtypes.as_dtype(tensor_type).base_dtype
  return gen_io_ops._restore_slice(
      file_pattern, tensor_name, shape_and_slice, base_type,
      preferred_shard, name=name)


@ops.RegisterShape("Restore")
def _RestoreShape(op):
  """Shape function for Restore op."""
  # Validate input shapes.
  unused_file_pattern = op.inputs[0].get_shape().merge_with(
      tensor_shape.scalar())
  unused_tensor_name = op.inputs[1].get_shape().merge_with(
      tensor_shape.scalar())
  return [tensor_shape.unknown_shape()]


@ops.RegisterShape("RestoreSlice")
def _RestoreSliceShape(op):
  """Shape function for RestoreSlice op."""
  # Validate input shapes.
  unused_file_pattern = op.inputs[0].get_shape().merge_with(
      tensor_shape.scalar())
  unused_tensor_name = op.inputs[1].get_shape().merge_with(
      tensor_shape.scalar())
  unused_shape_and_slice_shape = op.inputs[2].get_shape().merge_with(
      tensor_shape.scalar())
  # TODO(mrry): Attempt to parse the shape_and_slice value and use it
  # to form the shape of the output.
  return [tensor_shape.unknown_shape()]


@ops.RegisterShape("Save")
def _SaveShape(op):
  """Shape function for Save op."""
  # Validate input shapes.
  unused_filename = op.inputs[0].get_shape().merge_with(tensor_shape.scalar())
  data_count = len(op.inputs) - 2
  unused_tensor_names_shape = op.inputs[1].get_shape().merge_with(
      tensor_shape.vector(data_count))
  return []


@ops.RegisterShape("SaveSlices")
def _SaveSlicesShape(op):
  """Shape function for SaveSlices op."""
  # Validate input shapes.
  unused_filename = op.inputs[0].get_shape().merge_with(tensor_shape.scalar())
  data_count = len(op.inputs) - 3
  unused_tensor_names_shape = op.inputs[1].get_shape().merge_with(
      tensor_shape.vector(data_count))
  unused_shapes_and_slices_shape = op.inputs[2].get_shape().merge_with(
      tensor_shape.vector(data_count))
  # TODO(mrry): Attempt to parse the shapes_and_slices values and use
  # them to constrain the shape of the remaining inputs.
  return []


@ops.RegisterShape("ShardedFilename")
def _ShardedFilenameShape(op):
  """Shape function for ShardedFilename op."""
  # Validate input shapes.
  unused_basename_shape = op.inputs[0].get_shape().merge_with(
      tensor_shape.scalar())
  unused_shard_shape = op.inputs[1].get_shape().merge_with(
      tensor_shape.scalar())
  unused_num_shards_shape = op.inputs[2].get_shape().merge_with(
      tensor_shape.scalar())
  return [tensor_shape.scalar()]


@ops.RegisterShape("ShardedFilespec")
def _ShardedFilespecShape(op):
  """Shape function for ShardedFilespec op."""
  # Validate input shapes.
  unused_basename_shape = op.inputs[0].get_shape().merge_with(
      tensor_shape.scalar())
  unused_num_shards_shape = op.inputs[1].get_shape().merge_with(
      tensor_shape.scalar())
  return [tensor_shape.scalar()]


class ReaderBase(object):
  """Base class for different Reader types, that produce a record every step.

  Conceptually, Readers convert string 'work units' into records (key,
  value pairs).  Typically the 'work units' are filenames and the
  records are extracted from the contents of those files.  We want a
  single record produced per step, but a work unit can correspond to
  many records.

  Therefore we introduce some decoupling using a queue.  The queue
  contains the work units and the Reader dequeues from the queue when
  it is asked to produce a record (via Read()) but it has finished the
  last work unit.
  """

  def __init__(self, reader_ref, supports_serialize=False):
    """Creates a new ReaderBase.

    Args:
      reader_ref: The operation that implements the reader.
      supports_serialize: True if the reader implementation can
        serialize its state.
    """
    self._reader_ref = reader_ref
    self._supports_serialize = supports_serialize

  @property
  def reader_ref(self):
    """Op that implements the reader."""
    return self._reader_ref

  def read(self, queue, name=None):
    """Returns the next record (key, value pair) produced by a reader.

    Will dequeue a work unit from queue if necessary (e.g. when the
    Reader needs to start reading from a new file since it has
    finished with the previous file).

    Args:
      queue: A Queue or a mutable string Tensor representing a handle
        to a Queue, with string work items.
      name: A name for the operation (optional).

    Returns:
      A tuple of Tensors (key, value).
      key: A string scalar Tensor.
      value: A string scalar Tensor.
    """
    if isinstance(queue, ops.Tensor):
      queue_ref = queue
    else:
      queue_ref = queue.queue_ref
    return gen_io_ops._reader_read(self._reader_ref, queue_ref, name=name)

  def num_records_produced(self, name=None):
    """Returns the number of records this reader has produced.

    This is the same as the number of Read executions that have
    succeeded.

    Args:
      name: A name for the operation (optional).

    Returns:
      An int64 Tensor.

    """
    return gen_io_ops._reader_num_records_produced(self._reader_ref, name=name)

  def num_work_units_completed(self, name=None):
    """Returns the number of work units this reader has finished processing.

    Args:
      name: A name for the operation (optional).

    Returns:
      An int64 Tensor.
    """
    return gen_io_ops._reader_num_work_units_completed(self._reader_ref,
                                                       name=name)

  def serialize_state(self, name=None):
    """Produce a string tensor that encodes the state of a reader.

    Not all Readers support being serialized, so this can produce an
    Unimplemented error.

    Args:
      name: A name for the operation (optional).

    Returns:
      A string Tensor.
    """
    return gen_io_ops._reader_serialize_state(self._reader_ref, name=name)

  def restore_state(self, state, name=None):
    """Restore a reader to a previously saved state.

    Not all Readers support being restored, so this can produce an
    Unimplemented error.

    Args:
      state: A string Tensor.
        Result of a SerializeState of a Reader with matching type.
      name: A name for the operation (optional).

    Returns:
      The created Operation.
    """
    return gen_io_ops._reader_restore_state(self._reader_ref, state, name=name)

  @property
  def supports_serialize(self):
    """Whether the Reader implementation can serialize its state."""
    return self._supports_serialize

  def reset(self, name=None):
    """Restore a reader to its initial clean state.

    Args:
      name: A name for the operation (optional).

    Returns:
      The created Operation.
    """
    return gen_io_ops._reader_reset(self._reader_ref, name=name)


ops.NoGradient("ReaderRead")
ops.NoGradient("ReaderNumRecordsProduced")
ops.NoGradient("ReaderNumWorkUnitsCompleted")
ops.NoGradient("ReaderSerializeState")
ops.NoGradient("ReaderRestoreState")
ops.NoGradient("ReaderReset")


class WholeFileReader(ReaderBase):
  """A Reader that outputs the entire contents of a file as a value.

  To use, enqueue filenames in a Queue.  The output of Read will
  be a filename (key) and the contents of that file (value).

  See ReaderBase for supported methods.
  """

  def __init__(self, name=None):
    """Create a WholeFileReader.

    Args:
      name: A name for the operation (optional).
    """
    rr = gen_io_ops._whole_file_reader(name=name)
    super(WholeFileReader, self).__init__(rr, supports_serialize=True)


ops.NoGradient("WholeFileReader")


class TextLineReader(ReaderBase):
  """A Reader that outputs the lines of a file delimited by newlines.

  Newlines are stripped from the output.
  See ReaderBase for supported methods.
  """
  # TODO(josh11b): Support serializing and restoring state.

  def __init__(self, skip_header_lines=None, name=None):
    """Create a TextLineReader.

    Args:
      skip_header_lines: An optional int. Defaults to 0.  Number of lines
        to skip from the beginning of every file.
      name: A name for the operation (optional).
    """
    rr = gen_io_ops._text_line_reader(skip_header_lines=skip_header_lines,
                                      name=name)
    super(TextLineReader, self).__init__(rr)


ops.NoGradient("TextLineReader")


class FixedLengthRecordReader(ReaderBase):
  """A Reader that outputs fixed-length records from a file.

  See ReaderBase for supported methods.
  """
  # TODO(josh11b): Support serializing and restoring state.

  def __init__(self, record_bytes, header_bytes=None, footer_bytes=None,
               name=None):
    """Create a FixedLengthRecordReader.

    Args:
      record_bytes: An int.
      header_bytes: An optional int. Defaults to 0.
      footer_bytes: An optional int. Defaults to 0.
      name: A name for the operation (optional).
    """
    rr = gen_io_ops._fixed_length_record_reader(
        record_bytes=record_bytes, header_bytes=header_bytes,
        footer_bytes=footer_bytes, name=name)
    super(FixedLengthRecordReader, self).__init__(rr)


ops.NoGradient("FixedLengthRecordReader")


class TFRecordReader(ReaderBase):
  """A Reader that outputs the records from a TFRecords file.

  See ReaderBase for supported methods.
  """
  # TODO(josh11b): Support serializing and restoring state.

  def __init__(self, name=None):
    """Create a TFRecordReader.

    Args:
      name: A name for the operation (optional).
    """
    rr = gen_io_ops._tf_record_reader(name=name)
    super(TFRecordReader, self).__init__(rr)


ops.NoGradient("TFRecordReader")


class IdentityReader(ReaderBase):
  """A Reader that outputs the queued work as both the key and value.

  To use, enqueue strings in a Queue.  Read will take the front
  work string and output (work, work).

  See ReaderBase for supported methods.
  """

  def __init__(self, name=None):
    """Create a IdentityReader.

    Args:
      name: A name for the operation (optional).
    """
    rr = gen_io_ops._identity_reader(name=name)
    super(IdentityReader, self).__init__(rr, supports_serialize=True)


ops.NoGradient("IdentityReader")


ops.RegisterShape("FixedLengthRecordReader")(common_shapes.scalar_shape)
ops.RegisterShape("IdentityReader")(common_shapes.scalar_shape)
ops.RegisterShape("TextLineReader")(common_shapes.scalar_shape)
ops.RegisterShape("WholeFileReader")(common_shapes.scalar_shape)
ops.RegisterShape("TFRecordReader")(common_shapes.scalar_shape)


@ops.RegisterShape("ReaderNumRecordsProduced")
@ops.RegisterShape("ReaderNumWorkUnitsCompleted")
@ops.RegisterShape("ReaderSerializeState")
def _ReaderScalarShape(op):
  """Shape function for ops that transform a reader to a scalar."""
  unused_handle_shape = op.inputs[0].get_shape().merge_with(
      tensor_shape.scalar())
  return [tensor_shape.scalar()]


@ops.RegisterShape("ReaderRead")
def _ReaderReadShape(op):
  """Shape function for the ReaderBase.Read op."""
  unused_handle_shape = op.inputs[0].get_shape().merge_with(
      tensor_shape.scalar())
  unused_queue_shape = op.inputs[1].get_shape().merge_with(
      tensor_shape.scalar())
  return [tensor_shape.scalar(), tensor_shape.scalar()]


@ops.RegisterShape("ReaderReset")
def _ReaderResetShape(op):
  """Shape function for the ReaderBase.Reset op."""
  unused_handle_shape = op.inputs[0].get_shape().merge_with(
      tensor_shape.scalar())
  return []


@ops.RegisterShape("ReaderRestoreState")
def _ReaderRestoreStateShape(op):
  """Shape function for the ReaderBase.Restore op."""
  unused_handle_shape = op.inputs[0].get_shape().merge_with(
      tensor_shape.scalar())
  unused_state_shape = op.inputs[1].get_shape().merge_with(
      tensor_shape.scalar())
  return []


@ops.RegisterShape("ReadFile")
def _ReadFileShape(op):
  """Shape function for the ReadFile op."""
  return [op.inputs[0].get_shape().merge_with(tensor_shape.scalar())]


@ops.RegisterShape("MatchingFiles")
def _MatchingFilesShape(op):
  """Shape function for the MatchingFiles op."""
  unused_patern_shape = op.inputs[0].get_shape().merge_with(
      tensor_shape.scalar())
  return [tensor_shape.unknown_shape(ndims=1)]
