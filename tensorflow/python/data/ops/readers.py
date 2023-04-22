# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Python wrappers for reader Datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import convert
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export

_DEFAULT_READER_BUFFER_SIZE_BYTES = 256 * 1024  # 256 KB


def _normalise_fspath(path):
  """Convert pathlib-like objects to str (__fspath__ compatibility, PEP 519)."""
  return os.fspath(path) if isinstance(path, os.PathLike) else path


def _create_or_validate_filenames_dataset(filenames):
  """Creates (or validates) a dataset of filenames.

  Args:
    filenames: Either a list or dataset of filenames. If it is a list, it is
      convert to a dataset. If it is a dataset, its type and shape is validated.

  Returns:
    A dataset of filenames.
  """
  if isinstance(filenames, dataset_ops.DatasetV2):
    if dataset_ops.get_legacy_output_types(filenames) != dtypes.string:
      raise TypeError(
          "`filenames` must be a `tf.data.Dataset` of `tf.string` elements.")
    if not dataset_ops.get_legacy_output_shapes(filenames).is_compatible_with(
        tensor_shape.TensorShape([])):
      raise TypeError(
          "`filenames` must be a `tf.data.Dataset` of scalar `tf.string` "
          "elements.")
  else:
    filenames = nest.map_structure(_normalise_fspath, filenames)
    filenames = ops.convert_to_tensor(filenames, dtype_hint=dtypes.string)
    if filenames.dtype != dtypes.string:
      raise TypeError(
          "`filenames` must be a `tf.Tensor` of dtype `tf.string` dtype."
          " Got {}".format(filenames.dtype))
    filenames = array_ops.reshape(filenames, [-1], name="flat_filenames")
    filenames = dataset_ops.DatasetV2.from_tensor_slices(filenames)

  return filenames


def _create_dataset_reader(dataset_creator, filenames, num_parallel_reads=None):
  """Creates a dataset that reads the given files using the given reader.

  Args:
    dataset_creator: A function that takes in a single file name and returns a
      dataset.
    filenames: A `tf.data.Dataset` containing one or more filenames.
    num_parallel_reads: The number of parallel reads we should do.

  Returns:
    A `Dataset` that reads data from `filenames`.
  """

  def read_one_file(filename):
    filename = ops.convert_to_tensor(filename, dtypes.string, name="filename")
    return dataset_creator(filename)

  if num_parallel_reads is None:
    return filenames.flat_map(read_one_file)
  elif num_parallel_reads == dataset_ops.AUTOTUNE:
    return filenames.interleave(
        read_one_file, num_parallel_calls=num_parallel_reads)
  else:
    return ParallelInterleaveDataset(
        filenames,
        read_one_file,
        cycle_length=num_parallel_reads,
        block_length=1,
        sloppy=False,
        buffer_output_elements=None,
        prefetch_input_elements=None)


class _TextLineDataset(dataset_ops.DatasetSource):
  """A `Dataset` comprising records from one or more text files."""

  def __init__(self, filenames, compression_type=None, buffer_size=None):
    """Creates a `TextLineDataset`.

    Args:
      filenames: A `tf.string` tensor containing one or more filenames.
      compression_type: (Optional.) A `tf.string` scalar evaluating to one of
        `""` (no compression), `"ZLIB"`, or `"GZIP"`.
      buffer_size: (Optional.) A `tf.int64` scalar denoting the number of bytes
        to buffer. A value of 0 results in the default buffering values chosen
        based on the compression type.
    """
    self._filenames = filenames
    self._compression_type = convert.optional_param_to_tensor(
        "compression_type",
        compression_type,
        argument_default="",
        argument_dtype=dtypes.string)
    self._buffer_size = convert.optional_param_to_tensor(
        "buffer_size",
        buffer_size,
        argument_default=_DEFAULT_READER_BUFFER_SIZE_BYTES)
    variant_tensor = gen_dataset_ops.text_line_dataset(self._filenames,
                                                       self._compression_type,
                                                       self._buffer_size)
    super(_TextLineDataset, self).__init__(variant_tensor)

  @property
  def element_spec(self):
    return tensor_spec.TensorSpec([], dtypes.string)


@tf_export("data.TextLineDataset", v1=[])
class TextLineDatasetV2(dataset_ops.DatasetSource):
  r"""Creates a `Dataset` comprising lines from one or more text files.

  The `tf.data.TextLineDataset` loads text from text files and creates a dataset
  where each line of the files becomes an element of the dataset.

  For example, suppose we have 2 files "text_lines0.txt" and "text_lines1.txt"
  with the following lines:

  >>> with open('/tmp/text_lines0.txt', 'w') as f:
  ...   f.write('the cow\n')
  ...   f.write('jumped over\n')
  ...   f.write('the moon\n')
  >>> with open('/tmp/text_lines1.txt', 'w') as f:
  ...   f.write('jack and jill\n')
  ...   f.write('went up\n')
  ...   f.write('the hill\n')

  We can construct a TextLineDataset from them as follows:

  >>> dataset = tf.data.TextLineDataset(['/tmp/text_lines0.txt',
  ...                                    '/tmp/text_lines1.txt'])

  The elements of the dataset are expected to be:

  >>> for element in dataset.as_numpy_iterator():
  ...   print(element)
  b'the cow'
  b'jumped over'
  b'the moon'
  b'jack and jill'
  b'went up'
  b'the hill'
  """

  def __init__(self,
               filenames,
               compression_type=None,
               buffer_size=None,
               num_parallel_reads=None):
    r"""Creates a `TextLineDataset`.

    The elements of the dataset will be the lines of the input files, using
    the newline character '\n' to denote line splits. The newline characters
    will be stripped off of each element.

    Args:
      filenames: A `tf.string` tensor or `tf.data.Dataset` containing one or
        more filenames.
      compression_type: (Optional.) A `tf.string` scalar evaluating to one of
        `""` (no compression), `"ZLIB"`, or `"GZIP"`.
      buffer_size: (Optional.) A `tf.int64` scalar denoting the number of bytes
        to buffer. A value of 0 results in the default buffering values chosen
        based on the compression type.
      num_parallel_reads: (Optional.) A `tf.int64` scalar representing the
        number of files to read in parallel. If greater than one, the records of
        files read in parallel are outputted in an interleaved order. If your
        input pipeline is I/O bottlenecked, consider setting this parameter to a
        value greater than one to parallelize the I/O. If `None`, files will be
        read sequentially.
    """
    filenames = _create_or_validate_filenames_dataset(filenames)
    self._filenames = filenames
    self._compression_type = compression_type
    self._buffer_size = buffer_size

    def creator_fn(filename):
      return _TextLineDataset(filename, compression_type, buffer_size)

    self._impl = _create_dataset_reader(creator_fn, filenames,
                                        num_parallel_reads)
    variant_tensor = self._impl._variant_tensor  # pylint: disable=protected-access

    super(TextLineDatasetV2, self).__init__(variant_tensor)

  @property
  def element_spec(self):
    return tensor_spec.TensorSpec([], dtypes.string)


@tf_export(v1=["data.TextLineDataset"])
class TextLineDatasetV1(dataset_ops.DatasetV1Adapter):
  """A `Dataset` comprising lines from one or more text files."""

  def __init__(self,
               filenames,
               compression_type=None,
               buffer_size=None,
               num_parallel_reads=None):
    wrapped = TextLineDatasetV2(filenames, compression_type, buffer_size,
                                num_parallel_reads)
    super(TextLineDatasetV1, self).__init__(wrapped)

  __init__.__doc__ = TextLineDatasetV2.__init__.__doc__

  @property
  def _filenames(self):
    return self._dataset._filenames  # pylint: disable=protected-access

  @_filenames.setter
  def _filenames(self, value):
    self._dataset._filenames = value  # pylint: disable=protected-access


class _TFRecordDataset(dataset_ops.DatasetSource):
  """A `Dataset` comprising records from one or more TFRecord files."""

  def __init__(self, filenames, compression_type=None, buffer_size=None):
    """Creates a `TFRecordDataset`.

    Args:
      filenames: A `tf.string` tensor containing one or more filenames.
      compression_type: (Optional.) A `tf.string` scalar evaluating to one of
        `""` (no compression), `"ZLIB"`, or `"GZIP"`.
      buffer_size: (Optional.) A `tf.int64` scalar representing the number of
        bytes in the read buffer. 0 means no buffering.
    """
    self._filenames = filenames
    self._compression_type = convert.optional_param_to_tensor(
        "compression_type",
        compression_type,
        argument_default="",
        argument_dtype=dtypes.string)
    self._buffer_size = convert.optional_param_to_tensor(
        "buffer_size",
        buffer_size,
        argument_default=_DEFAULT_READER_BUFFER_SIZE_BYTES)
    variant_tensor = gen_dataset_ops.tf_record_dataset(self._filenames,
                                                       self._compression_type,
                                                       self._buffer_size)
    super(_TFRecordDataset, self).__init__(variant_tensor)

  @property
  def element_spec(self):
    return tensor_spec.TensorSpec([], dtypes.string)


class ParallelInterleaveDataset(dataset_ops.UnaryDataset):
  """A `Dataset` that maps a function over its input and flattens the result."""

  def __init__(self, input_dataset, map_func, cycle_length, block_length,
               sloppy, buffer_output_elements, prefetch_input_elements):
    """See `tf.data.experimental.parallel_interleave()` for details."""
    self._input_dataset = input_dataset
    self._map_func = dataset_ops.StructuredFunctionWrapper(
        map_func, self._transformation_name(), dataset=input_dataset)
    if not isinstance(self._map_func.output_structure, dataset_ops.DatasetSpec):
      raise TypeError("`map_func` must return a `Dataset` object.")
    self._element_spec = self._map_func.output_structure._element_spec  # pylint: disable=protected-access
    self._cycle_length = ops.convert_to_tensor(
        cycle_length, dtype=dtypes.int64, name="cycle_length")
    self._block_length = ops.convert_to_tensor(
        block_length, dtype=dtypes.int64, name="block_length")
    self._buffer_output_elements = convert.optional_param_to_tensor(
        "buffer_output_elements",
        buffer_output_elements,
        argument_default=2 * block_length)
    self._prefetch_input_elements = convert.optional_param_to_tensor(
        "prefetch_input_elements",
        prefetch_input_elements,
        argument_default=2 * cycle_length)
    if sloppy is None:
      self._deterministic = "default"
    elif sloppy:
      self._deterministic = "false"
    else:
      self._deterministic = "true"
    variant_tensor = ged_ops.legacy_parallel_interleave_dataset_v2(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._map_func.function.captured_inputs,
        self._cycle_length,
        self._block_length,
        self._buffer_output_elements,
        self._prefetch_input_elements,
        f=self._map_func.function,
        deterministic=self._deterministic,
        **self._flat_structure)
    super(ParallelInterleaveDataset, self).__init__(input_dataset,
                                                    variant_tensor)

  def _functions(self):
    return [self._map_func]

  @property
  def element_spec(self):
    return self._element_spec

  def _transformation_name(self):
    return "tf.data.experimental.parallel_interleave()"


@tf_export("data.TFRecordDataset", v1=[])
class TFRecordDatasetV2(dataset_ops.DatasetV2):
  """A `Dataset` comprising records from one or more TFRecord files.

  This dataset loads TFRecords from the files as bytes, exactly as they were
  written.`TFRecordDataset` does not do any parsing or decoding on its own.
  Parsing and decoding can be done by applying `Dataset.map` transformations
  after the `TFRecordDataset`.

  A minimal example is given below:

  >>> import tempfile
  >>> example_path = os.path.join(tempfile.gettempdir(), "example.tfrecords")
  >>> np.random.seed(0)

  >>> # Write the records to a file.
  ... with tf.io.TFRecordWriter(example_path) as file_writer:
  ...   for _ in range(4):
  ...     x, y = np.random.random(), np.random.random()
  ...
  ...     record_bytes = tf.train.Example(features=tf.train.Features(feature={
  ...         "x": tf.train.Feature(float_list=tf.train.FloatList(value=[x])),
  ...         "y": tf.train.Feature(float_list=tf.train.FloatList(value=[y])),
  ...     })).SerializeToString()
  ...     file_writer.write(record_bytes)

  >>> # Read the data back out.
  >>> def decode_fn(record_bytes):
  ...   return tf.io.parse_single_example(
  ...       # Data
  ...       record_bytes,
  ...
  ...       # Schema
  ...       {"x": tf.io.FixedLenFeature([], dtype=tf.float32),
  ...        "y": tf.io.FixedLenFeature([], dtype=tf.float32)}
  ...   )

  >>> for batch in tf.data.TFRecordDataset([example_path]).map(decode_fn):
  ...   print("x = {x:.4f},  y = {y:.4f}".format(**batch))
  x = 0.5488,  y = 0.7152
  x = 0.6028,  y = 0.5449
  x = 0.4237,  y = 0.6459
  x = 0.4376,  y = 0.8918
  """

  def __init__(self,
               filenames,
               compression_type=None,
               buffer_size=None,
               num_parallel_reads=None):
    """Creates a `TFRecordDataset` to read one or more TFRecord files.

    Each element of the dataset will contain a single TFRecord.

    Args:
      filenames: A `tf.string` tensor or `tf.data.Dataset` containing one or
        more filenames.
      compression_type: (Optional.) A `tf.string` scalar evaluating to one of
        `""` (no compression), `"ZLIB"`, or `"GZIP"`.
      buffer_size: (Optional.) A `tf.int64` scalar representing the number of
        bytes in the read buffer. If your input pipeline is I/O bottlenecked,
        consider setting this parameter to a value 1-100 MBs. If `None`, a
        sensible default for both local and remote file systems is used.
      num_parallel_reads: (Optional.) A `tf.int64` scalar representing the
        number of files to read in parallel. If greater than one, the records of
        files read in parallel are outputted in an interleaved order. If your
        input pipeline is I/O bottlenecked, consider setting this parameter to a
        value greater than one to parallelize the I/O. If `None`, files will be
        read sequentially.

    Raises:
      TypeError: If any argument does not have the expected type.
      ValueError: If any argument does not have the expected shape.
    """
    filenames = _create_or_validate_filenames_dataset(filenames)

    self._filenames = filenames
    self._compression_type = compression_type
    self._buffer_size = buffer_size
    self._num_parallel_reads = num_parallel_reads

    def creator_fn(filename):
      return _TFRecordDataset(filename, compression_type, buffer_size)

    self._impl = _create_dataset_reader(creator_fn, filenames,
                                        num_parallel_reads)
    variant_tensor = self._impl._variant_tensor  # pylint: disable=protected-access
    super(TFRecordDatasetV2, self).__init__(variant_tensor)

  def _clone(self,
             filenames=None,
             compression_type=None,
             buffer_size=None,
             num_parallel_reads=None):
    return TFRecordDatasetV2(filenames or self._filenames, compression_type or
                             self._compression_type, buffer_size or
                             self._buffer_size, num_parallel_reads or
                             self._num_parallel_reads)

  def _inputs(self):
    return self._impl._inputs()  # pylint: disable=protected-access

  @property
  def element_spec(self):
    return tensor_spec.TensorSpec([], dtypes.string)


@tf_export(v1=["data.TFRecordDataset"])
class TFRecordDatasetV1(dataset_ops.DatasetV1Adapter):
  """A `Dataset` comprising records from one or more TFRecord files."""

  def __init__(self,
               filenames,
               compression_type=None,
               buffer_size=None,
               num_parallel_reads=None):
    wrapped = TFRecordDatasetV2(filenames, compression_type, buffer_size,
                                num_parallel_reads)
    super(TFRecordDatasetV1, self).__init__(wrapped)

  __init__.__doc__ = TFRecordDatasetV2.__init__.__doc__

  def _clone(self,
             filenames=None,
             compression_type=None,
             buffer_size=None,
             num_parallel_reads=None):
    # pylint: disable=protected-access
    return TFRecordDatasetV1(
        filenames or self._dataset._filenames, compression_type or
        self._dataset._compression_type, buffer_size or
        self._dataset._buffer_size, num_parallel_reads or
        self._dataset._num_parallel_reads)

  @property
  def _filenames(self):
    return self._dataset._filenames  # pylint: disable=protected-access

  @_filenames.setter
  def _filenames(self, value):
    self._dataset._filenames = value  # pylint: disable=protected-access


class _FixedLengthRecordDataset(dataset_ops.DatasetSource):
  """A `Dataset` of fixed-length records from one or more binary files."""

  def __init__(self,
               filenames,
               record_bytes,
               header_bytes=None,
               footer_bytes=None,
               buffer_size=None,
               compression_type=None):
    """Creates a `FixedLengthRecordDataset`.

    Args:
      filenames: A `tf.string` tensor containing one or more filenames.
      record_bytes: A `tf.int64` scalar representing the number of bytes in each
        record.
      header_bytes: (Optional.) A `tf.int64` scalar representing the number of
        bytes to skip at the start of a file.
      footer_bytes: (Optional.) A `tf.int64` scalar representing the number of
        bytes to ignore at the end of a file.
      buffer_size: (Optional.) A `tf.int64` scalar representing the number of
        bytes to buffer when reading.
      compression_type: (Optional.) A `tf.string` scalar evaluating to one of
        `""` (no compression), `"ZLIB"`, or `"GZIP"`.
    """
    self._filenames = filenames
    self._record_bytes = ops.convert_to_tensor(
        record_bytes, dtype=dtypes.int64, name="record_bytes")
    self._header_bytes = convert.optional_param_to_tensor(
        "header_bytes", header_bytes)
    self._footer_bytes = convert.optional_param_to_tensor(
        "footer_bytes", footer_bytes)
    self._buffer_size = convert.optional_param_to_tensor(
        "buffer_size", buffer_size, _DEFAULT_READER_BUFFER_SIZE_BYTES)
    self._compression_type = convert.optional_param_to_tensor(
        "compression_type",
        compression_type,
        argument_default="",
        argument_dtype=dtypes.string)
    variant_tensor = gen_dataset_ops.fixed_length_record_dataset_v2(
        self._filenames, self._header_bytes, self._record_bytes,
        self._footer_bytes, self._buffer_size, self._compression_type)
    super(_FixedLengthRecordDataset, self).__init__(variant_tensor)

  @property
  def element_spec(self):
    return tensor_spec.TensorSpec([], dtypes.string)


@tf_export("data.FixedLengthRecordDataset", v1=[])
class FixedLengthRecordDatasetV2(dataset_ops.DatasetSource):
  """A `Dataset` of fixed-length records from one or more binary files.

  The `tf.data.FixedLengthRecordDataset` reads fixed length records from binary
  files and creates a dataset where each record becomes an element of the
  dataset. The binary files can have a fixed length header and a fixed length
  footer, which will both be skipped.

  For example, suppose we have 2 files "fixed_length0.bin" and
  "fixed_length1.bin" with the following content:

  >>> with open('/tmp/fixed_length0.bin', 'wb') as f:
  ...   f.write(b'HEADER012345FOOTER')
  >>> with open('/tmp/fixed_length1.bin', 'wb') as f:
  ...   f.write(b'HEADER6789abFOOTER')

  We can construct a `FixedLengthRecordDataset` from them as follows:

  >>> dataset1 = tf.data.FixedLengthRecordDataset(
  ...     filenames=['/tmp/fixed_length0.bin', '/tmp/fixed_length1.bin'],
  ...     record_bytes=2, header_bytes=6, footer_bytes=6)

  The elements of the dataset are:

  >>> for element in dataset1.as_numpy_iterator():
  ...   print(element)
  b'01'
  b'23'
  b'45'
  b'67'
  b'89'
  b'ab'
  """

  def __init__(self,
               filenames,
               record_bytes,
               header_bytes=None,
               footer_bytes=None,
               buffer_size=None,
               compression_type=None,
               num_parallel_reads=None):
    """Creates a `FixedLengthRecordDataset`.

    Args:
      filenames: A `tf.string` tensor or `tf.data.Dataset` containing one or
        more filenames.
      record_bytes: A `tf.int64` scalar representing the number of bytes in each
        record.
      header_bytes: (Optional.) A `tf.int64` scalar representing the number of
        bytes to skip at the start of a file.
      footer_bytes: (Optional.) A `tf.int64` scalar representing the number of
        bytes to ignore at the end of a file.
      buffer_size: (Optional.) A `tf.int64` scalar representing the number of
        bytes to buffer when reading.
      compression_type: (Optional.) A `tf.string` scalar evaluating to one of
        `""` (no compression), `"ZLIB"`, or `"GZIP"`.
      num_parallel_reads: (Optional.) A `tf.int64` scalar representing the
        number of files to read in parallel. If greater than one, the records of
        files read in parallel are outputted in an interleaved order. If your
        input pipeline is I/O bottlenecked, consider setting this parameter to a
        value greater than one to parallelize the I/O. If `None`, files will be
        read sequentially.
    """
    filenames = _create_or_validate_filenames_dataset(filenames)

    self._filenames = filenames
    self._record_bytes = record_bytes
    self._header_bytes = header_bytes
    self._footer_bytes = footer_bytes
    self._buffer_size = buffer_size
    self._compression_type = compression_type

    def creator_fn(filename):
      return _FixedLengthRecordDataset(filename, record_bytes, header_bytes,
                                       footer_bytes, buffer_size,
                                       compression_type)

    self._impl = _create_dataset_reader(creator_fn, filenames,
                                        num_parallel_reads)
    variant_tensor = self._impl._variant_tensor  # pylint: disable=protected-access
    super(FixedLengthRecordDatasetV2, self).__init__(variant_tensor)

  @property
  def element_spec(self):
    return tensor_spec.TensorSpec([], dtypes.string)


@tf_export(v1=["data.FixedLengthRecordDataset"])
class FixedLengthRecordDatasetV1(dataset_ops.DatasetV1Adapter):
  """A `Dataset` of fixed-length records from one or more binary files."""

  def __init__(self,
               filenames,
               record_bytes,
               header_bytes=None,
               footer_bytes=None,
               buffer_size=None,
               compression_type=None,
               num_parallel_reads=None):
    wrapped = FixedLengthRecordDatasetV2(filenames, record_bytes, header_bytes,
                                         footer_bytes, buffer_size,
                                         compression_type, num_parallel_reads)
    super(FixedLengthRecordDatasetV1, self).__init__(wrapped)

  __init__.__doc__ = FixedLengthRecordDatasetV2.__init__.__doc__

  @property
  def _filenames(self):
    return self._dataset._filenames  # pylint: disable=protected-access

  @_filenames.setter
  def _filenames(self, value):
    self._dataset._filenames = value  # pylint: disable=protected-access


if tf2.enabled():
  FixedLengthRecordDataset = FixedLengthRecordDatasetV2
  TFRecordDataset = TFRecordDatasetV2
  TextLineDataset = TextLineDatasetV2
else:
  FixedLengthRecordDataset = FixedLengthRecordDatasetV1
  TFRecordDataset = TFRecordDatasetV1
  TextLineDataset = TextLineDatasetV1
