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

"""For reading and writing TFRecords files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import errors
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


@tf_export(
    v1=["io.TFRecordCompressionType", "python_io.TFRecordCompressionType"])
@deprecation.deprecated_endpoints("io.TFRecordCompressionType",
                                  "python_io.TFRecordCompressionType")
class TFRecordCompressionType(object):
  """The type of compression for the record."""
  NONE = 0
  ZLIB = 1
  GZIP = 2


@tf_export(
    "io.TFRecordOptions",
    v1=["io.TFRecordOptions", "python_io.TFRecordOptions"])
@deprecation.deprecated_endpoints("python_io.TFRecordOptions")
class TFRecordOptions(object):
  """Options used for manipulating TFRecord files."""
  compression_type_map = {
      TFRecordCompressionType.ZLIB: "ZLIB",
      TFRecordCompressionType.GZIP: "GZIP",
      TFRecordCompressionType.NONE: ""
  }

  def __init__(self,
               compression_type=None,
               flush_mode=None,
               input_buffer_size=None,
               output_buffer_size=None,
               window_bits=None,
               compression_level=None,
               compression_method=None,
               mem_level=None,
               compression_strategy=None):
    # pylint: disable=line-too-long
    """Creates a `TFRecordOptions` instance.

    Options only effect TFRecordWriter when compression_type is not `None`.
    Documentation, details, and defaults can be found in
    [`zlib_compression_options.h`](https://www.tensorflow.org/code/tensorflow/core/lib/io/zlib_compression_options.h)
    and in the [zlib manual](http://www.zlib.net/manual.html).
    Leaving an option as `None` allows C++ to set a reasonable default.

    Args:
      compression_type: `"GZIP"`, "ZLIB" or `"NONE"`.
      flush_mode: flush mode or `None`, Default: Z_NO_FLUSH.
      input_buffer_size: int or `None`.
      output_buffer_size: int or `None`.
      window_bits: int or `None`.
      compression_level: 0 to 9, or `None`.
      compression_method: compression method or `None`.
      mem_level: 1 to 9, or `None`.
      compression_strategy: strategy or `None`. Default: Z_DEFAULT_STRATEGY.

    Returns:
      A `TFRecordOptions` object.

    Raises:
      ValueError: If compression_type is invalid.
    """
    # pylint: enable=line-too-long
    # Check compression_type is valid, but for backwards compatibility don't
    # immediately convert to a string.
    self.get_compression_type_string(compression_type)
    self.compression_type = compression_type
    self.flush_mode = flush_mode
    self.input_buffer_size = input_buffer_size
    self.output_buffer_size = output_buffer_size
    self.window_bits = window_bits
    self.compression_level = compression_level
    self.compression_method = compression_method
    self.mem_level = mem_level
    self.compression_strategy = compression_strategy

  @classmethod
  def get_compression_type_string(cls, options):
    """Convert various option types to a unified string.

    Args:
      options: `TFRecordOption`, `TFRecordCompressionType`, or string.

    Returns:
      Compression type as string (e.g. `'ZLIB'`, `'GZIP'`, or `''`).

    Raises:
      ValueError: If compression_type is invalid.
    """
    if not options:
      return ""
    elif isinstance(options, TFRecordOptions):
      return cls.get_compression_type_string(options.compression_type)
    elif isinstance(options, TFRecordCompressionType):
      return cls.compression_type_map[options]
    elif options in TFRecordOptions.compression_type_map:
      return cls.compression_type_map[options]
    elif options in TFRecordOptions.compression_type_map.values():
      return options
    else:
      raise ValueError('Not a valid compression_type: "{}"'.format(options))

  def _as_record_writer_options(self):
    """Convert to RecordWriterOptions for use with PyRecordWriter."""
    options = pywrap_tensorflow.RecordWriterOptions_CreateRecordWriterOptions(
        compat.as_bytes(
            self.get_compression_type_string(self.compression_type)))

    if self.flush_mode is not None:
      options.zlib_options.flush_mode = self.flush_mode
    if self.input_buffer_size is not None:
      options.zlib_options.input_buffer_size = self.input_buffer_size
    if self.output_buffer_size is not None:
      options.zlib_options.output_buffer_size = self.output_buffer_size
    if self.window_bits is not None:
      options.zlib_options.window_bits = self.window_bits
    if self.compression_level is not None:
      options.zlib_options.compression_level = self.compression_level
    if self.compression_method is not None:
      options.zlib_options.compression_method = self.compression_method
    if self.mem_level is not None:
      options.zlib_options.mem_level = self.mem_level
    if self.compression_strategy is not None:
      options.zlib_options.compression_strategy = self.compression_strategy
    return options


@tf_export(v1=["io.tf_record_iterator", "python_io.tf_record_iterator"])
@deprecation.deprecated(
    date=None,
    instructions=("Use eager execution and: \n"
                  "`tf.data.TFRecordDataset(path)`"))
def tf_record_iterator(path, options=None):
  """An iterator that read the records from a TFRecords file.

  Args:
    path: The path to the TFRecords file.
    options: (optional) A TFRecordOptions object.

  Yields:
    Strings.

  Raises:
    IOError: If `path` cannot be opened for reading.
  """
  compression_type = TFRecordOptions.get_compression_type_string(options)
  with errors.raise_exception_on_not_ok_status() as status:
    reader = pywrap_tensorflow.PyRecordReader_New(
        compat.as_bytes(path), 0, compat.as_bytes(compression_type), status)

  if reader is None:
    raise IOError("Could not open %s." % path)
  try:
    while True:
      try:
        reader.GetNext()
      except errors.OutOfRangeError:
        break
      yield reader.record()
  finally:
    reader.Close()


@tf_export(
    "io.TFRecordWriter", v1=["io.TFRecordWriter", "python_io.TFRecordWriter"])
@deprecation.deprecated_endpoints("python_io.TFRecordWriter")
class TFRecordWriter(object):
  """A class to write records to a TFRecords file.

  This class implements `__enter__` and `__exit__`, and can be used
  in `with` blocks like a normal file.
  """

  # TODO(josh11b): Support appending?
  def __init__(self, path, options=None):
    """Opens file `path` and creates a `TFRecordWriter` writing to it.

    Args:
      path: The path to the TFRecords file.
      options: (optional) String specifying compression type,
          `TFRecordCompressionType`, or `TFRecordOptions` object.

    Raises:
      IOError: If `path` cannot be opened for writing.
      ValueError: If valid compression_type can't be determined from `options`.
    """
    if not isinstance(options, TFRecordOptions):
      options = TFRecordOptions(compression_type=options)

    with errors.raise_exception_on_not_ok_status() as status:
      # pylint: disable=protected-access
      self._writer = pywrap_tensorflow.PyRecordWriter_New(
          compat.as_bytes(path), options._as_record_writer_options(), status)
      # pylint: enable=protected-access

  def __enter__(self):
    """Enter a `with` block."""
    return self

  def __exit__(self, unused_type, unused_value, unused_traceback):
    """Exit a `with` block, closing the file."""
    self.close()

  def write(self, record):
    """Write a string record to the file.

    Args:
      record: str
    """
    with errors.raise_exception_on_not_ok_status() as status:
      self._writer.WriteRecord(record, status)

  def flush(self):
    """Flush the file."""
    with errors.raise_exception_on_not_ok_status() as status:
      self._writer.Flush(status)

  def close(self):
    """Close the file."""
    with errors.raise_exception_on_not_ok_status() as status:
      self._writer.Close(status)
