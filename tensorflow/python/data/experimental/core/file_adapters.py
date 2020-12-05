# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
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

"""Adapters for file formats."""

import abc
import enum
import os

from typing import Any, ClassVar, Iterable

import tensorflow.compat.v2 as tf

from tensorflow.data.experimental.core.utils import type_utils

try:
  import riegeli  # pylint: disable=g-import-not-at-top
except ImportError:
  riegeli = Any

try:
  from riegeli.tensorflow.ops import riegeli_dataset_ops as riegeli_tf  # pylint: disable=g-import-not-at-top
except ImportError:
  riegeli_tf = Any


class FileFormat(enum.Enum):
  """Format of the record files.

  The values of the enumeration are used as filename endings/suffix.
  """
  TFRECORD = 'tfrecord'
  RIEGELI = 'riegeli'


DEFAULT_FILE_FORMAT = FileFormat.TFRECORD


class FileAdapter(abc.ABC):
  """Interface for Adapter objects which read and write examples in a format."""

  PATH_SUFFIX = ClassVar[str]

  @abc.abstractclassmethod
  def make_tf_data(cls,
                   filename: type_utils.PathLike,
                   buffer_size: tf.int64) -> tf.data.Dataset:
    """Returns TensorFlow Dataset comprising given record file."""
    raise NotImplementedError()

  @abc.abstractclassmethod
  def write_examples(cls, path: type_utils.PathLike, iterator: Iterable[bytes]):
    """Write examples from given iterator in given path."""
    raise NotImplementedError()


class TfRecordFileAdapter(FileAdapter):
  """File adapter for TFRecord file format."""

  FILE_SUFFIX = 'tfrecord'

  @classmethod
  def make_tf_data(cls,
                   filename: type_utils.PathLike,
                   buffer_size: tf.int64) -> tf.data.Dataset:
    """Returns TensorFlow Dataset comprising given record file."""
    return tf.data.TFRecordDataset(filename, buffer_size=buffer_size)

  @classmethod
  def write_examples(cls, path: type_utils.PathLike, iterator: Iterable[bytes]):
    """Write examples from given iterator in given path."""
    with tf.io.TFRecordWriter(os.fspath(path)) as writer:
      for serialized_example in iterator:
        writer.write(serialized_example)
      writer.flush()


class RiegeliFileAdapter(FileAdapter):
  """File adapter for Riegeli file format."""

  FILE_SUFFIX = 'riegeli'

  @classmethod
  def make_tf_data(cls,
                   filename: type_utils.PathLike,
                   buffer_size: tf.int64) -> tf.data.Dataset:
    """Returns TensorFlow Dataset comprising given record file."""
    return riegeli_tf.RiegeliDataset(filename, buffer_size=buffer_size)

  @classmethod
  def write_examples(cls, path: type_utils.PathLike, iterator: Iterable[bytes]):
    """Write examples from given iterator in given path."""
    with riegeli.RecordWriter(
        tf.io.gfile.GFile(os.fspath(path), 'wb'),
        options='transpose') as writer:
      writer.write_records(records=iterator)


# Create a mapping from FileFormat -> FileAdapter.
ADAPTER_FOR_FORMAT = {
    FileFormat.RIEGELI: RiegeliFileAdapter,
    FileFormat.TFRECORD: TfRecordFileAdapter
}


def is_example_file(filename: str) -> bool:
  """Whether the given filename is a record file."""
  return any(
      f'.{adapter.FILE_SUFFIX}' in filename
      for adapter in ADAPTER_FOR_FORMAT.values()
  )
