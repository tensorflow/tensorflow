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


class TFRecordCompressionType(object):
  """The type of compression for the record."""
  NONE = 0
  ZLIB = 1
  GZIP = 2


# NOTE(vrv): This will eventually be converted into a proto.  to match
# the interface used by the C++ RecordWriter.
class TFRecordOptions(object):
  """Options used for manipulating TFRecord files."""
  compression_type_map = {
      TFRecordCompressionType.ZLIB: "ZLIB",
      TFRecordCompressionType.GZIP: "GZIP",
      TFRecordCompressionType.NONE: ""
  }

  def __init__(self, compression_type):
    self.compression_type = compression_type

  @classmethod
  def get_compression_type_string(cls, options):
    if not options:
      return ""
    return cls.compression_type_map[options.compression_type]


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
  while True:
    try:
      with errors.raise_exception_on_not_ok_status() as status:
        reader.GetNext(status)
    except errors.OutOfRangeError:
      break
    yield reader.record()
  reader.Close()


class TFRecordWriter(object):
  """A class to write records to a TFRecords file.

  This class implements `__enter__` and `__exit__`, and can be used
  in `with` blocks like a normal file.

  @@__init__
  @@write
  @@close
  """

  # TODO(josh11b): Support appending?
  def __init__(self, path, options=None):
    """Opens file `path` and creates a `TFRecordWriter` writing to it.

    Args:
      path: The path to the TFRecords file.
      options: (optional) A TFRecordOptions object.

    Raises:
      IOError: If `path` cannot be opened for writing.
    """
    compression_type = TFRecordOptions.get_compression_type_string(options)

    with errors.raise_exception_on_not_ok_status() as status:
      self._writer = pywrap_tensorflow.PyRecordWriter_New(
          compat.as_bytes(path), compat.as_bytes(compression_type), status)

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
    self._writer.WriteRecord(record)

  def close(self):
    """Close the file."""
    self._writer.Close()
