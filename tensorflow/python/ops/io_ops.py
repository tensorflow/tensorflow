# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Inputs and Readers.

See the [Inputs and
Readers](https://tensorflow.org/api_guides/python/io_ops) guide.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import python_io
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_io_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_io_ops import *
from tensorflow.python.util.tf_export import tf_export
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
    return gen_io_ops.save(filename, tensor_names, tensors, name=name)
  else:
    return gen_io_ops.save_slices(filename, tensor_names, tensor_slices,
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
  return gen_io_ops.restore_slice(
      file_pattern, tensor_name, shape_and_slice, base_type,
      preferred_shard, name=name)


@tf_export("ReaderBase")
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

  @compatibility(eager)
  Readers are not compatible with eager execution. Instead, please
  use `tf.data` to get data into your model.
  @end_compatibility
  """

  def __init__(self, reader_ref, supports_serialize=False):
    """Creates a new ReaderBase.

    Args:
      reader_ref: The operation that implements the reader.
      supports_serialize: True if the reader implementation can
        serialize its state.

    Raises:
      RuntimeError: If eager execution is enabled.
    """
    if context.executing_eagerly():
      raise RuntimeError(
          "Readers are not supported when eager execution is enabled. "
          "Instead, please use tf.data to get data into your model.")

    self._reader_ref = reader_ref
    self._supports_serialize = supports_serialize

  @property
  def reader_ref(self):
    """Op that implements the reader."""
    return self._reader_ref

  def read(self, queue, name=None):
    """Returns the next record (key, value) pair produced by a reader.

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
    if self._reader_ref.dtype == dtypes.resource:
      return gen_io_ops.reader_read_v2(self._reader_ref, queue_ref, name=name)
    else:
      # For compatibility with pre-resource queues, create a ref(string) tensor
      # which can be looked up as the same queue by a resource manager.
      old_queue_op = gen_data_flow_ops.fake_queue(queue_ref)
      return gen_io_ops.reader_read(self._reader_ref, old_queue_op, name=name)

  def read_up_to(self, queue, num_records,  # pylint: disable=invalid-name
                 name=None):
    """Returns up to num_records (key, value) pairs produced by a reader.

    Will dequeue a work unit from queue if necessary (e.g., when the
    Reader needs to start reading from a new file since it has
    finished with the previous file).
    It may return less than num_records even before the last batch.

    Args:
      queue: A Queue or a mutable string Tensor representing a handle
        to a Queue, with string work items.
      num_records: Number of records to read.
      name: A name for the operation (optional).

    Returns:
      A tuple of Tensors (keys, values).
      keys: A 1-D string Tensor.
      values: A 1-D string Tensor.
    """
    if isinstance(queue, ops.Tensor):
      queue_ref = queue
    else:
      queue_ref = queue.queue_ref
    if self._reader_ref.dtype == dtypes.resource:
      return gen_io_ops.reader_read_up_to_v2(self._reader_ref,
                                             queue_ref,
                                             num_records,
                                             name=name)
    else:
      # For compatibility with pre-resource queues, create a ref(string) tensor
      # which can be looked up as the same queue by a resource manager.
      old_queue_op = gen_data_flow_ops.fake_queue(queue_ref)
      return gen_io_ops.reader_read_up_to(self._reader_ref,
                                          old_queue_op,
                                          num_records,
                                          name=name)

  def num_records_produced(self, name=None):
    """Returns the number of records this reader has produced.

    This is the same as the number of Read executions that have
    succeeded.

    Args:
      name: A name for the operation (optional).

    Returns:
      An int64 Tensor.

    """
    if self._reader_ref.dtype == dtypes.resource:
      return gen_io_ops.reader_num_records_produced_v2(self._reader_ref,
                                                       name=name)
    else:
      return gen_io_ops.reader_num_records_produced(self._reader_ref,
                                                    name=name)

  def num_work_units_completed(self, name=None):
    """Returns the number of work units this reader has finished processing.

    Args:
      name: A name for the operation (optional).

    Returns:
      An int64 Tensor.
    """
    if self._reader_ref.dtype == dtypes.resource:
      return gen_io_ops.reader_num_work_units_completed_v2(self._reader_ref,
                                                           name=name)
    else:
      return gen_io_ops.reader_num_work_units_completed(self._reader_ref,
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
    if self._reader_ref.dtype == dtypes.resource:
      return gen_io_ops.reader_serialize_state_v2(self._reader_ref, name=name)
    else:
      return gen_io_ops.reader_serialize_state(self._reader_ref, name=name)

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
    if self._reader_ref.dtype == dtypes.resource:
      return gen_io_ops.reader_restore_state_v2(
          self._reader_ref, state, name=name)
    else:
      return gen_io_ops.reader_restore_state(self._reader_ref, state, name=name)

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
    if self._reader_ref.dtype == dtypes.resource:
      return gen_io_ops.reader_reset_v2(self._reader_ref, name=name)
    else:
      return gen_io_ops.reader_reset(self._reader_ref, name=name)


ops.NotDifferentiable("ReaderRead")
ops.NotDifferentiable("ReaderReadUpTo")
ops.NotDifferentiable("ReaderNumRecordsProduced")
ops.NotDifferentiable("ReaderNumWorkUnitsCompleted")
ops.NotDifferentiable("ReaderSerializeState")
ops.NotDifferentiable("ReaderRestoreState")
ops.NotDifferentiable("ReaderReset")


@tf_export("WholeFileReader")
class WholeFileReader(ReaderBase):
  """A Reader that outputs the entire contents of a file as a value.

  To use, enqueue filenames in a Queue.  The output of Read will
  be a filename (key) and the contents of that file (value).

  See ReaderBase for supported methods.

  @compatibility(eager)
  Readers are not compatible with eager execution. Instead, please
  use `tf.data` to get data into your model.
  @end_compatibility
  """

  def __init__(self, name=None):
    """Create a WholeFileReader.

    Args:
      name: A name for the operation (optional).
    """
    rr = gen_io_ops.whole_file_reader_v2(name=name)
    super(WholeFileReader, self).__init__(rr, supports_serialize=True)


ops.NotDifferentiable("WholeFileReader")


@tf_export("TextLineReader")
class TextLineReader(ReaderBase):
  """A Reader that outputs the lines of a file delimited by newlines.

  Newlines are stripped from the output.
  See ReaderBase for supported methods.

  @compatibility(eager)
  Readers are not compatible with eager execution. Instead, please
  use `tf.data` to get data into your model.
  @end_compatibility
  """
  # TODO(josh11b): Support serializing and restoring state.

  def __init__(self, skip_header_lines=None, name=None):
    """Create a TextLineReader.

    Args:
      skip_header_lines: An optional int. Defaults to 0.  Number of lines
        to skip from the beginning of every file.
      name: A name for the operation (optional).
    """
    rr = gen_io_ops.text_line_reader_v2(skip_header_lines=skip_header_lines,
                                        name=name)
    super(TextLineReader, self).__init__(rr)


ops.NotDifferentiable("TextLineReader")


@tf_export("FixedLengthRecordReader")
class FixedLengthRecordReader(ReaderBase):
  """A Reader that outputs fixed-length records from a file.

  See ReaderBase for supported methods.

  @compatibility(eager)
  Readers are not compatible with eager execution. Instead, please
  use `tf.data` to get data into your model.
  @end_compatibility
  """
  # TODO(josh11b): Support serializing and restoring state.

  def __init__(self,
               record_bytes,
               header_bytes=None,
               footer_bytes=None,
               hop_bytes=None,
               name=None,
               encoding=None):
    """Create a FixedLengthRecordReader.

    Args:
      record_bytes: An int.
      header_bytes: An optional int. Defaults to 0.
      footer_bytes: An optional int. Defaults to 0.
      hop_bytes: An optional int. Defaults to 0.
      name: A name for the operation (optional).
      encoding: The type of encoding for the file. Defaults to none.
    """
    rr = gen_io_ops.fixed_length_record_reader_v2(
        record_bytes=record_bytes,
        header_bytes=header_bytes,
        footer_bytes=footer_bytes,
        hop_bytes=hop_bytes,
        encoding=encoding,
        name=name)
    super(FixedLengthRecordReader, self).__init__(rr)


ops.NotDifferentiable("FixedLengthRecordReader")


@tf_export("TFRecordReader")
class TFRecordReader(ReaderBase):
  """A Reader that outputs the records from a TFRecords file.

  See ReaderBase for supported methods.

  @compatibility(eager)
  Readers are not compatible with eager execution. Instead, please
  use `tf.data` to get data into your model.
  @end_compatibility
  """
  # TODO(josh11b): Support serializing and restoring state.

  def __init__(self, name=None, options=None):
    """Create a TFRecordReader.

    Args:
      name: A name for the operation (optional).
      options: A TFRecordOptions object (optional).
    """
    compression_type = python_io.TFRecordOptions.get_compression_type_string(
        options)

    rr = gen_io_ops.tf_record_reader_v2(
        name=name, compression_type=compression_type)
    super(TFRecordReader, self).__init__(rr)


ops.NotDifferentiable("TFRecordReader")


@tf_export("LMDBReader")
class LMDBReader(ReaderBase):
  """A Reader that outputs the records from a LMDB file.

  See ReaderBase for supported methods.

  @compatibility(eager)
  Readers are not compatible with eager execution. Instead, please
  use `tf.data` to get data into your model.
  @end_compatibility
  """
  def __init__(self, name=None, options=None):
    """Create a LMDBReader.

    Args:
      name: A name for the operation (optional).
      options: A LMDBRecordOptions object (optional).
    """
    rr = gen_io_ops.lmdb_reader(name=name)
    super(LMDBReader, self).__init__(rr)


ops.NotDifferentiable("LMDBReader")


@tf_export("IdentityReader")
class IdentityReader(ReaderBase):
  """A Reader that outputs the queued work as both the key and value.

  To use, enqueue strings in a Queue.  Read will take the front
  work string and output (work, work).

  See ReaderBase for supported methods.

  @compatibility(eager)
  Readers are not compatible with eager execution. Instead, please
  use `tf.data` to get data into your model.
  @end_compatibility
  """

  def __init__(self, name=None):
    """Create a IdentityReader.

    Args:
      name: A name for the operation (optional).
    """
    rr = gen_io_ops.identity_reader_v2(name=name)
    super(IdentityReader, self).__init__(rr, supports_serialize=True)


ops.NotDifferentiable("IdentityReader")
