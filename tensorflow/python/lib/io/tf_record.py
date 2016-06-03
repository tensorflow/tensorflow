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
from tensorflow.python.util import compat


def tf_record_iterator(path):
  """An iterator that read the records from a TFRecords file.

  Args:
    path: The path to the TFRecords file.

  Yields:
    Strings.

  Raises:
    IOError: If `path` cannot be opened for reading.
  """
  reader = pywrap_tensorflow.PyRecordReader_New(compat.as_bytes(path), 0)
  if reader is None:
    raise IOError("Could not open %s." % path)
  while reader.GetNext():
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
  def __init__(self, path):
    """Opens file `path` and creates a `TFRecordWriter` writing to it.

    Args:
      path: The path to the TFRecords file.

    Raises:
      IOError: If `path` cannot be opened for writing.
    """
    self._writer = pywrap_tensorflow.PyRecordWriter_New(compat.as_bytes(path))
    if self._writer is None:
      raise IOError("Could not write to %s." % path)

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
