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

import collections
import csv
import functools
import gzip

import numpy as np

from tensorflow.python import tf2
from tensorflow.python.data.experimental.ops import error_ops
from tensorflow.python.data.experimental.ops import parsing_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers as core_readers
from tensorflow.python.data.util import convert
from tensorflow.python.data.util import nest
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util.tf_export import tf_export

_ACCEPTABLE_CSV_TYPES = (dtypes.float32, dtypes.float64, dtypes.int32,
                         dtypes.int64, dtypes.string)


def _is_valid_int32(str_val):
  try:
    # Checks equality to prevent int32 overflow
    return dtypes.int32.as_numpy_dtype(str_val) == dtypes.int64.as_numpy_dtype(
        str_val)
  except (ValueError, OverflowError):
    return False


def _is_valid_int64(str_val):
  try:
    dtypes.int64.as_numpy_dtype(str_val)
    return True
  except (ValueError, OverflowError):
    return False


def _is_valid_float(str_val, float_dtype):
  try:
    return float_dtype.as_numpy_dtype(str_val) < np.inf
  except ValueError:
    return False


def _infer_type(str_val, na_value, prev_type):
  """Given a string, infers its tensor type.

  Infers the type of a value by picking the least 'permissive' type possible,
  while still allowing the previous type inference for this column to be valid.

  Args:
    str_val: String value to infer the type of.
    na_value: Additional string to recognize as a NA/NaN CSV value.
    prev_type: Type previously inferred based on values of this column that
      we've seen up till now.
  Returns:
    Inferred dtype.
  """
  if str_val in ("", na_value):
    # If the field is null, it gives no extra information about its type
    return prev_type

  type_list = [
      dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64, dtypes.string
  ]  # list of types to try, ordered from least permissive to most

  type_functions = [
      _is_valid_int32,
      _is_valid_int64,
      lambda str_val: _is_valid_float(str_val, dtypes.float32),
      lambda str_val: _is_valid_float(str_val, dtypes.float64),
      lambda str_val: True,
  ]  # Corresponding list of validation functions

  for i in range(len(type_list)):
    validation_fn = type_functions[i]
    if validation_fn(str_val) and (prev_type is None or
                                   prev_type in type_list[:i + 1]):
      return type_list[i]


def _next_csv_row(filenames, num_cols, field_delim, use_quote_delim, header,
                  file_io_fn):
  """Generator that yields rows of CSV file(s) in order."""
  for fn in filenames:
    with file_io_fn(fn) as f:
      rdr = csv.reader(
          f,
          delimiter=field_delim,
          quoting=csv.QUOTE_MINIMAL if use_quote_delim else csv.QUOTE_NONE)
      if header:
        next(rdr)  # Skip header lines

      for csv_row in rdr:
        if len(csv_row) != num_cols:
          raise ValueError(
              "Problem inferring types: CSV row has different number of fields "
              "than expected.")
        yield csv_row


def _infer_column_defaults(filenames, num_cols, field_delim, use_quote_delim,
                           na_value, header, num_rows_for_inference,
                           select_columns, file_io_fn):
  """Infers column types from the first N valid CSV records of files."""
  if select_columns is None:
    select_columns = range(num_cols)
  inferred_types = [None] * len(select_columns)

  for i, csv_row in enumerate(
      _next_csv_row(filenames, num_cols, field_delim, use_quote_delim, header,
                    file_io_fn)):
    if num_rows_for_inference is not None and i >= num_rows_for_inference:
      break

    for j, col_index in enumerate(select_columns):
      inferred_types[j] = _infer_type(csv_row[col_index], na_value,
                                      inferred_types[j])

  # Replace None's with a default type
  inferred_types = [t or dtypes.string for t in inferred_types]
  # Default to 0 or '' for null values
  return [
      constant_op.constant([0 if t is not dtypes.string else ""], dtype=t)
      for t in inferred_types
  ]


def _infer_column_names(filenames, field_delim, use_quote_delim, file_io_fn):
  """Infers column names from first rows of files."""
  csv_kwargs = {
      "delimiter": field_delim,
      "quoting": csv.QUOTE_MINIMAL if use_quote_delim else csv.QUOTE_NONE
  }
  with file_io_fn(filenames[0]) as f:
    try:
      column_names = next(csv.reader(f, **csv_kwargs))
    except StopIteration:
      raise ValueError(("Received StopIteration when reading the header line "
                        "of %s.  Empty file?") % filenames[0])

  for name in filenames[1:]:
    with file_io_fn(name) as f:
      try:
        if next(csv.reader(f, **csv_kwargs)) != column_names:
          raise ValueError(
              "Files have different column names in the header row.")
      except StopIteration:
        raise ValueError(("Received StopIteration when reading the header line "
                          "of %s.  Empty file?") % filenames[0])
  return column_names


def _get_sorted_col_indices(select_columns, column_names):
  """Transforms select_columns argument into sorted column indices."""
  names_to_indices = {n: i for i, n in enumerate(column_names)}
  num_cols = len(column_names)

  results = []
  for v in select_columns:
    # If value is already an int, check if it's valid.
    if isinstance(v, int):
      if v < 0 or v >= num_cols:
        raise ValueError(
            "Column index %d specified in select_columns out of valid range." %
            v)
      results.append(v)
    # Otherwise, check that it's a valid column name and convert to the
    # the relevant column index.
    elif v not in names_to_indices:
      raise ValueError(
          "Value '%s' specified in select_columns not a valid column index or "
          "name." % v)
    else:
      results.append(names_to_indices[v])

  # Sort and ensure there are no duplicates
  results = sorted(set(results))
  if len(results) != len(select_columns):
    raise ValueError("select_columns contains duplicate columns")
  return results


def _maybe_shuffle_and_repeat(
    dataset, num_epochs, shuffle, shuffle_buffer_size, shuffle_seed):
  """Optionally shuffle and repeat dataset, as requested."""
  if shuffle:
    dataset = dataset.shuffle(shuffle_buffer_size, shuffle_seed)
  if num_epochs != 1:
    dataset = dataset.repeat(num_epochs)
  return dataset


def make_tf_record_dataset(file_pattern,
                           batch_size,
                           parser_fn=None,
                           num_epochs=None,
                           shuffle=True,
                           shuffle_buffer_size=None,
                           shuffle_seed=None,
                           prefetch_buffer_size=None,
                           num_parallel_reads=None,
                           num_parallel_parser_calls=None,
                           drop_final_batch=False):
  """Reads and optionally parses TFRecord files into a dataset.

  Provides common functionality such as batching, optional parsing, shuffling,
  and performant defaults.

  Args:
    file_pattern: List of files or patterns of TFRecord file paths.
      See `tf.io.gfile.glob` for pattern rules.
    batch_size: An int representing the number of records to combine
      in a single batch.
    parser_fn: (Optional.) A function accepting string input to parse
      and process the record contents. This function must map records
      to components of a fixed shape, so they may be batched. By
      default, uses the record contents unmodified.
    num_epochs: (Optional.) An int specifying the number of times this
      dataset is repeated.  If None (the default), cycles through the
      dataset forever.
    shuffle: (Optional.) A bool that indicates whether the input
      should be shuffled. Defaults to `True`.
    shuffle_buffer_size: (Optional.) Buffer size to use for
      shuffling. A large buffer size ensures better shuffling, but
      increases memory usage and startup time.
    shuffle_seed: (Optional.) Randomization seed to use for shuffling.
    prefetch_buffer_size: (Optional.) An int specifying the number of
      feature batches to prefetch for performance improvement.
      Defaults to auto-tune. Set to 0 to disable prefetching.
    num_parallel_reads: (Optional.) Number of threads used to read
      records from files. By default or if set to a value >1, the
      results will be interleaved. Defaults to `24`.
    num_parallel_parser_calls: (Optional.) Number of parallel
      records to parse in parallel. Defaults to `batch_size`.
    drop_final_batch: (Optional.) Whether the last batch should be
      dropped in case its size is smaller than `batch_size`; the
      default behavior is not to drop the smaller batch.

  Returns:
    A dataset, where each element matches the output of `parser_fn`
    except it will have an additional leading `batch-size` dimension,
    or a `batch_size`-length 1-D tensor of strings if `parser_fn` is
    unspecified.
  """
  if num_parallel_reads is None:
    # NOTE: We considered auto-tuning this value, but there is a concern
    # that this affects the mixing of records from different files, which
    # could affect training convergence/accuracy, so we are defaulting to
    # a constant for now.
    num_parallel_reads = 24

  if num_parallel_parser_calls is None:
    # TODO(josh11b): if num_parallel_parser_calls is None, use some function
    # of num cores instead of `batch_size`.
    num_parallel_parser_calls = batch_size

  if prefetch_buffer_size is None:
    prefetch_buffer_size = dataset_ops.AUTOTUNE

  files = dataset_ops.Dataset.list_files(
      file_pattern, shuffle=shuffle, seed=shuffle_seed)

  dataset = core_readers.TFRecordDataset(
      files, num_parallel_reads=num_parallel_reads)

  if shuffle_buffer_size is None:
    # TODO(josh11b): Auto-tune this value when not specified
    shuffle_buffer_size = 10000
  dataset = _maybe_shuffle_and_repeat(
      dataset, num_epochs, shuffle, shuffle_buffer_size, shuffle_seed)

  # NOTE(mrry): We set `drop_final_batch=True` when `num_epochs is None` to
  # improve the shape inference, because it makes the batch dimension static.
  # It is safe to do this because in that case we are repeating the input
  # indefinitely, and all batches will be full-sized.
  drop_final_batch = drop_final_batch or num_epochs is None

  if parser_fn is None:
    dataset = dataset.batch(batch_size, drop_remainder=drop_final_batch)
  else:
    dataset = dataset.map(
        parser_fn, num_parallel_calls=num_parallel_parser_calls)
    dataset = dataset.batch(batch_size, drop_remainder=drop_final_batch)

  if prefetch_buffer_size == 0:
    return dataset
  else:
    return dataset.prefetch(buffer_size=prefetch_buffer_size)


@tf_export("data.experimental.make_csv_dataset", v1=[])
def make_csv_dataset_v2(
    file_pattern,
    batch_size,
    column_names=None,
    column_defaults=None,
    label_name=None,
    select_columns=None,
    field_delim=",",
    use_quote_delim=True,
    na_value="",
    header=True,
    num_epochs=None,
    shuffle=True,
    shuffle_buffer_size=10000,
    shuffle_seed=None,
    prefetch_buffer_size=None,
    num_parallel_reads=None,
    sloppy=False,
    num_rows_for_inference=100,
    compression_type=None,
    ignore_errors=False,
):
  """Reads CSV files into a dataset.

  Reads CSV files into a dataset, where each element is a (features, labels)
  tuple that corresponds to a batch of CSV rows. The features dictionary
  maps feature column names to `Tensor`s containing the corresponding
  feature data, and labels is a `Tensor` containing the batch's label data.

  Args:
    file_pattern: List of files or patterns of file paths containing CSV
      records. See `tf.io.gfile.glob` for pattern rules.
    batch_size: An int representing the number of records to combine
      in a single batch.
    column_names: An optional list of strings that corresponds to the CSV
      columns, in order. One per column of the input record. If this is not
      provided, infers the column names from the first row of the records.
      These names will be the keys of the features dict of each dataset element.
    column_defaults: A optional list of default values for the CSV fields. One
      item per selected column of the input record. Each item in the list is
      either a valid CSV dtype (float32, float64, int32, int64, or string), or a
      `Tensor` with one of the aforementioned types. The tensor can either be
      a scalar default value (if the column is optional), or an empty tensor (if
      the column is required). If a dtype is provided instead of a tensor, the
      column is also treated as required. If this list is not provided, tries
      to infer types based on reading the first num_rows_for_inference rows of
      files specified, and assumes all columns are optional, defaulting to `0`
      for numeric values and `""` for string values. If both this and
      `select_columns` are specified, these must have the same lengths, and
      `column_defaults` is assumed to be sorted in order of increasing column
      index.
    label_name: A optional string corresponding to the label column. If
      provided, the data for this column is returned as a separate `Tensor` from
      the features dictionary, so that the dataset complies with the format
      expected by a `tf.Estimator.train` or `tf.Estimator.evaluate` input
      function.
    select_columns: An optional list of integer indices or string column
      names, that specifies a subset of columns of CSV data to select. If
      column names are provided, these must correspond to names provided in
      `column_names` or inferred from the file header lines. When this argument
      is specified, only a subset of CSV columns will be parsed and returned,
      corresponding to the columns specified. Using this results in faster
      parsing and lower memory usage. If both this and `column_defaults` are
      specified, these must have the same lengths, and `column_defaults` is
      assumed to be sorted in order of increasing column index.
    field_delim: An optional `string`. Defaults to `","`. Char delimiter to
      separate fields in a record.
    use_quote_delim: An optional bool. Defaults to `True`. If false, treats
      double quotation marks as regular characters inside of the string fields.
    na_value: Additional string to recognize as NA/NaN.
    header: A bool that indicates whether the first rows of provided CSV files
      correspond to header lines with column names, and should not be included
      in the data.
    num_epochs: An int specifying the number of times this dataset is repeated.
      If None, cycles through the dataset forever.
    shuffle: A bool that indicates whether the input should be shuffled.
    shuffle_buffer_size: Buffer size to use for shuffling. A large buffer size
      ensures better shuffling, but increases memory usage and startup time.
    shuffle_seed: Randomization seed to use for shuffling.
    prefetch_buffer_size: An int specifying the number of feature
      batches to prefetch for performance improvement. Recommended value is the
      number of batches consumed per training step. Defaults to auto-tune.
    num_parallel_reads: Number of threads used to read CSV records from files.
      If >1, the results will be interleaved. Defaults to `1`.
    sloppy: If `True`, reading performance will be improved at
      the cost of non-deterministic ordering. If `False`, the order of elements
      produced is deterministic prior to shuffling (elements are still
      randomized if `shuffle=True`. Note that if the seed is set, then order
      of elements after shuffling is deterministic). Defaults to `False`.
    num_rows_for_inference: Number of rows of a file to use for type inference
      if record_defaults is not provided. If None, reads all the rows of all
      the files. Defaults to 100.
    compression_type: (Optional.) A `tf.string` scalar evaluating to one of
      `""` (no compression), `"ZLIB"`, or `"GZIP"`. Defaults to no compression.
    ignore_errors: (Optional.) If `True`, ignores errors with CSV file parsing,
      such as malformed data or empty lines, and moves on to the next valid
      CSV record. Otherwise, the dataset raises an error and stops processing
      when encountering any invalid records. Defaults to `False`.

  Returns:
    A dataset, where each element is a (features, labels) tuple that corresponds
    to a batch of `batch_size` CSV rows. The features dictionary maps feature
    column names to `Tensor`s containing the corresponding column data, and
    labels is a `Tensor` containing the column data for the label column
    specified by `label_name`.

  Raises:
    ValueError: If any of the arguments is malformed.
  """
  if num_parallel_reads is None:
    num_parallel_reads = 1

  if prefetch_buffer_size is None:
    prefetch_buffer_size = dataset_ops.AUTOTUNE

  # Create dataset of all matching filenames
  filenames = _get_file_names(file_pattern, False)
  dataset = dataset_ops.Dataset.from_tensor_slices(filenames)
  if shuffle:
    dataset = dataset.shuffle(len(filenames), shuffle_seed)

  # Clean arguments; figure out column names and defaults
  if column_names is None or column_defaults is None:
    # Find out which io function to open the file
    file_io_fn = lambda filename: file_io.FileIO(filename, "r")
    if compression_type is not None:
      compression_type_value = tensor_util.constant_value(compression_type)
      if compression_type_value is None:
        raise ValueError("Received unknown compression_type")
      if compression_type_value == "GZIP":
        file_io_fn = lambda filename: gzip.open(filename, "rt")
      elif compression_type_value == "ZLIB":
        raise ValueError(
            "compression_type (%s) is not supported for probing columns" %
            compression_type)
      elif compression_type_value != "":
        raise ValueError("compression_type (%s) is not supported" %
                         compression_type)
  if column_names is None:
    if not header:
      raise ValueError("Cannot infer column names without a header line.")
    # If column names are not provided, infer from the header lines
    column_names = _infer_column_names(filenames, field_delim, use_quote_delim,
                                       file_io_fn)
  if len(column_names) != len(set(column_names)):
    raise ValueError("Cannot have duplicate column names.")

  if select_columns is not None:
    select_columns = _get_sorted_col_indices(select_columns, column_names)

  if column_defaults is not None:
    column_defaults = [
        constant_op.constant([], dtype=x)
        if not tensor_util.is_tensor(x) and x in _ACCEPTABLE_CSV_TYPES else x
        for x in column_defaults
    ]
  else:
    # If column defaults are not provided, infer from records at graph
    # construction time
    column_defaults = _infer_column_defaults(filenames, len(column_names),
                                             field_delim, use_quote_delim,
                                             na_value, header,
                                             num_rows_for_inference,
                                             select_columns, file_io_fn)

  if select_columns is not None and len(column_defaults) != len(select_columns):
    raise ValueError(
        "If specified, column_defaults and select_columns must have same "
        "length."
    )
  if select_columns is not None and len(column_names) > len(select_columns):
    # Pick the relevant subset of column names
    column_names = [column_names[i] for i in select_columns]

  if label_name is not None and label_name not in column_names:
    raise ValueError("`label_name` provided must be one of the columns.")

  def filename_to_dataset(filename):
    dataset = CsvDataset(
        filename,
        record_defaults=column_defaults,
        field_delim=field_delim,
        use_quote_delim=use_quote_delim,
        na_value=na_value,
        select_cols=select_columns,
        header=header,
        compression_type=compression_type
    )
    if ignore_errors:
      dataset = dataset.apply(error_ops.ignore_errors())
    return dataset

  def map_fn(*columns):
    """Organizes columns into a features dictionary.

    Args:
      *columns: list of `Tensor`s corresponding to one csv record.
    Returns:
      An OrderedDict of feature names to values for that particular record. If
      label_name is provided, extracts the label feature to be returned as the
      second element of the tuple.
    """
    features = collections.OrderedDict(zip(column_names, columns))
    if label_name is not None:
      label = features.pop(label_name)
      return features, label
    return features

  if num_parallel_reads == dataset_ops.AUTOTUNE:
    dataset = dataset.interleave(
        filename_to_dataset, num_parallel_calls=num_parallel_reads)
    options = dataset_ops.Options()
    options.experimental_deterministic = not sloppy
    dataset = dataset.with_options(options)
  else:
    # Read files sequentially (if num_parallel_reads=1) or in parallel
    def apply_fn(dataset):
      return core_readers.ParallelInterleaveDataset(
          dataset,
          filename_to_dataset,
          cycle_length=num_parallel_reads,
          block_length=1,
          sloppy=sloppy,
          buffer_output_elements=None,
          prefetch_input_elements=None)

    dataset = dataset.apply(apply_fn)

  dataset = _maybe_shuffle_and_repeat(
      dataset, num_epochs, shuffle, shuffle_buffer_size, shuffle_seed)

  # Apply batch before map for perf, because map has high overhead relative
  # to the size of the computation in each map.
  # NOTE(mrry): We set `drop_remainder=True` when `num_epochs is None` to
  # improve the shape inference, because it makes the batch dimension static.
  # It is safe to do this because in that case we are repeating the input
  # indefinitely, and all batches will be full-sized.
  dataset = dataset.batch(batch_size=batch_size,
                          drop_remainder=num_epochs is None)
  dataset = dataset_ops.MapDataset(
      dataset, map_fn, use_inter_op_parallelism=False)
  dataset = dataset.prefetch(prefetch_buffer_size)

  return dataset


@tf_export(v1=["data.experimental.make_csv_dataset"])
def make_csv_dataset_v1(
    file_pattern,
    batch_size,
    column_names=None,
    column_defaults=None,
    label_name=None,
    select_columns=None,
    field_delim=",",
    use_quote_delim=True,
    na_value="",
    header=True,
    num_epochs=None,
    shuffle=True,
    shuffle_buffer_size=10000,
    shuffle_seed=None,
    prefetch_buffer_size=None,
    num_parallel_reads=None,
    sloppy=False,
    num_rows_for_inference=100,
    compression_type=None,
    ignore_errors=False,
):  # pylint: disable=missing-docstring
  return dataset_ops.DatasetV1Adapter(make_csv_dataset_v2(
      file_pattern, batch_size, column_names, column_defaults, label_name,
      select_columns, field_delim, use_quote_delim, na_value, header,
      num_epochs, shuffle, shuffle_buffer_size, shuffle_seed,
      prefetch_buffer_size, num_parallel_reads, sloppy, num_rows_for_inference,
      compression_type, ignore_errors))
make_csv_dataset_v1.__doc__ = make_csv_dataset_v2.__doc__


_DEFAULT_READER_BUFFER_SIZE_BYTES = 4 * 1024 * 1024  # 4 MB


@tf_export("data.experimental.CsvDataset", v1=[])
class CsvDatasetV2(dataset_ops.DatasetSource):
  """A Dataset comprising lines from one or more CSV files."""

  def __init__(self,
               filenames,
               record_defaults,
               compression_type=None,
               buffer_size=None,
               header=False,
               field_delim=",",
               use_quote_delim=True,
               na_value="",
               select_cols=None):
    """Creates a `CsvDataset` by reading and decoding CSV files.

    The elements of this dataset correspond to records from the file(s).
    RFC 4180 format is expected for CSV files
    (https://tools.ietf.org/html/rfc4180)
    Note that we allow leading and trailing spaces with int or float field.


    For example, suppose we have a file 'my_file0.csv' with four CSV columns of
    different data types:
    ```
    abcdefg,4.28E10,5.55E6,12
    hijklmn,-5.3E14,,2
    ```

    We can construct a CsvDataset from it as follows:

    ```python
     dataset = tf.data.experimental.CsvDataset(
        "my_file*.csv",
        [tf.float32,  # Required field, use dtype or empty tensor
         tf.constant([0.0], dtype=tf.float32),  # Optional field, default to 0.0
         tf.int32,  # Required field, use dtype or empty tensor
         ],
        select_cols=[1,2,3]  # Only parse last three columns
    )
    ```

    The expected output of its iterations is:

    ```python
    for element in dataset:
      print(element)

    >> (4.28e10, 5.55e6, 12)
    >> (-5.3e14, 0.0, 2)
    ```

    Args:
      filenames: A `tf.string` tensor containing one or more filenames.
      record_defaults: A list of default values for the CSV fields. Each item in
        the list is either a valid CSV `DType` (float32, float64, int32, int64,
        string), or a `Tensor` object with one of the above types. One per
        column of CSV data, with either a scalar `Tensor` default value for the
        column if it is optional, or `DType` or empty `Tensor` if required. If
        both this and `select_columns` are specified, these must have the same
        lengths, and `column_defaults` is assumed to be sorted in order of
        increasing column index.
      compression_type: (Optional.) A `tf.string` scalar evaluating to one of
        `""` (no compression), `"ZLIB"`, or `"GZIP"`. Defaults to no
        compression.
      buffer_size: (Optional.) A `tf.int64` scalar denoting the number of bytes
        to buffer while reading files. Defaults to 4MB.
      header: (Optional.) A `tf.bool` scalar indicating whether the CSV file(s)
        have header line(s) that should be skipped when parsing. Defaults to
        `False`.
      field_delim: (Optional.) A `tf.string` scalar containing the delimiter
        character that separates fields in a record. Defaults to `","`.
      use_quote_delim: (Optional.) A `tf.bool` scalar. If `False`, treats
        double quotation marks as regular characters inside of string fields
        (ignoring RFC 4180, Section 2, Bullet 5). Defaults to `True`.
      na_value: (Optional.) A `tf.string` scalar indicating a value that will
        be treated as NA/NaN.
      select_cols: (Optional.) A sorted list of column indices to select from
        the input data. If specified, only this subset of columns will be
        parsed. Defaults to parsing all columns.
    """
    self._filenames = ops.convert_to_tensor(
        filenames, dtype=dtypes.string, name="filenames")
    self._compression_type = convert.optional_param_to_tensor(
        "compression_type",
        compression_type,
        argument_default="",
        argument_dtype=dtypes.string)
    record_defaults = [
        constant_op.constant([], dtype=x)
        if not tensor_util.is_tensor(x) and x in _ACCEPTABLE_CSV_TYPES else x
        for x in record_defaults
    ]
    self._record_defaults = ops.convert_n_to_tensor(
        record_defaults, name="record_defaults")
    self._buffer_size = convert.optional_param_to_tensor(
        "buffer_size", buffer_size, _DEFAULT_READER_BUFFER_SIZE_BYTES)
    self._header = ops.convert_to_tensor(
        header, dtype=dtypes.bool, name="header")
    self._field_delim = ops.convert_to_tensor(
        field_delim, dtype=dtypes.string, name="field_delim")
    self._use_quote_delim = ops.convert_to_tensor(
        use_quote_delim, dtype=dtypes.bool, name="use_quote_delim")
    self._na_value = ops.convert_to_tensor(
        na_value, dtype=dtypes.string, name="na_value")
    self._select_cols = convert.optional_param_to_tensor(
        "select_cols",
        select_cols,
        argument_default=[],
        argument_dtype=dtypes.int64,
    )
    self._element_spec = tuple(
        tensor_spec.TensorSpec([], d.dtype) for d in self._record_defaults)
    variant_tensor = gen_experimental_dataset_ops.csv_dataset(
        filenames=self._filenames,
        record_defaults=self._record_defaults,
        buffer_size=self._buffer_size,
        header=self._header,
        output_shapes=self._flat_shapes,
        field_delim=self._field_delim,
        use_quote_delim=self._use_quote_delim,
        na_value=self._na_value,
        select_cols=self._select_cols,
        compression_type=self._compression_type)
    super(CsvDatasetV2, self).__init__(variant_tensor)

  @property
  def element_spec(self):
    return self._element_spec


@tf_export(v1=["data.experimental.CsvDataset"])
class CsvDatasetV1(dataset_ops.DatasetV1Adapter):
  """A Dataset comprising lines from one or more CSV files."""

  @functools.wraps(CsvDatasetV2.__init__)
  def __init__(self,
               filenames,
               record_defaults,
               compression_type=None,
               buffer_size=None,
               header=False,
               field_delim=",",
               use_quote_delim=True,
               na_value="",
               select_cols=None):
    wrapped = CsvDatasetV2(filenames, record_defaults, compression_type,
                           buffer_size, header, field_delim, use_quote_delim,
                           na_value, select_cols)
    super(CsvDatasetV1, self).__init__(wrapped)


@tf_export("data.experimental.make_batched_features_dataset", v1=[])
def make_batched_features_dataset_v2(file_pattern,
                                     batch_size,
                                     features,
                                     reader=None,
                                     label_key=None,
                                     reader_args=None,
                                     num_epochs=None,
                                     shuffle=True,
                                     shuffle_buffer_size=10000,
                                     shuffle_seed=None,
                                     prefetch_buffer_size=None,
                                     reader_num_threads=None,
                                     parser_num_threads=None,
                                     sloppy_ordering=False,
                                     drop_final_batch=False):
  """Returns a `Dataset` of feature dictionaries from `Example` protos.

  If label_key argument is provided, returns a `Dataset` of tuple
  comprising of feature dictionaries and label.

  Example:

  ```
  serialized_examples = [
    features {
      feature { key: "age" value { int64_list { value: [ 0 ] } } }
      feature { key: "gender" value { bytes_list { value: [ "f" ] } } }
      feature { key: "kws" value { bytes_list { value: [ "code", "art" ] } } }
    },
    features {
      feature { key: "age" value { int64_list { value: [] } } }
      feature { key: "gender" value { bytes_list { value: [ "f" ] } } }
      feature { key: "kws" value { bytes_list { value: [ "sports" ] } } }
    }
  ]
  ```

  We can use arguments:

  ```
  features: {
    "age": FixedLenFeature([], dtype=tf.int64, default_value=-1),
    "gender": FixedLenFeature([], dtype=tf.string),
    "kws": VarLenFeature(dtype=tf.string),
  }
  ```

  And the expected output is:

  ```python
  {
    "age": [[0], [-1]],
    "gender": [["f"], ["f"]],
    "kws": SparseTensor(
      indices=[[0, 0], [0, 1], [1, 0]],
      values=["code", "art", "sports"]
      dense_shape=[2, 2]),
  }
  ```

  Args:
    file_pattern: List of files or patterns of file paths containing
      `Example` records. See `tf.io.gfile.glob` for pattern rules.
    batch_size: An int representing the number of records to combine
      in a single batch.
    features: A `dict` mapping feature keys to `FixedLenFeature` or
      `VarLenFeature` values. See `tf.io.parse_example`.
    reader: A function or class that can be
      called with a `filenames` tensor and (optional) `reader_args` and returns
      a `Dataset` of `Example` tensors. Defaults to `tf.data.TFRecordDataset`.
    label_key: (Optional) A string corresponding to the key labels are stored in
      `tf.Examples`. If provided, it must be one of the `features` key,
      otherwise results in `ValueError`.
    reader_args: Additional arguments to pass to the reader class.
    num_epochs: Integer specifying the number of times to read through the
      dataset. If None, cycles through the dataset forever. Defaults to `None`.
    shuffle: A boolean, indicates whether the input should be shuffled. Defaults
      to `True`.
    shuffle_buffer_size: Buffer size of the ShuffleDataset. A large capacity
      ensures better shuffling but would increase memory usage and startup time.
    shuffle_seed: Randomization seed to use for shuffling.
    prefetch_buffer_size: Number of feature batches to prefetch in order to
      improve performance. Recommended value is the number of batches consumed
      per training step. Defaults to auto-tune.
    reader_num_threads: Number of threads used to read `Example` records. If >1,
      the results will be interleaved. Defaults to `1`.
    parser_num_threads: Number of threads to use for parsing `Example` tensors
      into a dictionary of `Feature` tensors. Defaults to `2`.
    sloppy_ordering: If `True`, reading performance will be improved at
      the cost of non-deterministic ordering. If `False`, the order of elements
      produced is deterministic prior to shuffling (elements are still
      randomized if `shuffle=True`. Note that if the seed is set, then order
      of elements after shuffling is deterministic). Defaults to `False`.
    drop_final_batch: If `True`, and the batch size does not evenly divide the
      input dataset size, the final smaller batch will be dropped. Defaults to
      `False`.

  Returns:
    A dataset of `dict` elements, (or a tuple of `dict` elements and label).
    Each `dict` maps feature keys to `Tensor` or `SparseTensor` objects.

  Raises:
    TypeError: If `reader` is of the wrong type.
    ValueError: If `label_key` is not one of the `features` keys.
  """
  if reader is None:
    reader = core_readers.TFRecordDataset

  if reader_num_threads is None:
    reader_num_threads = 1
  if parser_num_threads is None:
    parser_num_threads = 2
  if prefetch_buffer_size is None:
    prefetch_buffer_size = dataset_ops.AUTOTUNE

  # Create dataset of all matching filenames
  dataset = dataset_ops.Dataset.list_files(
      file_pattern, shuffle=shuffle, seed=shuffle_seed)

  if isinstance(reader, type) and issubclass(reader, io_ops.ReaderBase):
    raise TypeError("The `reader` argument must return a `Dataset` object. "
                    "`tf.ReaderBase` subclasses are not supported. For "
                    "example, pass `tf.data.TFRecordDataset` instead of "
                    "`tf.TFRecordReader`.")

  # Read `Example` records from files as tensor objects.
  if reader_args is None:
    reader_args = []

  if reader_num_threads == dataset_ops.AUTOTUNE:
    dataset = dataset.interleave(
        lambda filename: reader(filename, *reader_args),
        num_parallel_calls=reader_num_threads)
    options = dataset_ops.Options()
    options.experimental_deterministic = not sloppy_ordering
    dataset = dataset.with_options(options)
  else:
    # Read files sequentially (if reader_num_threads=1) or in parallel
    def apply_fn(dataset):
      return core_readers.ParallelInterleaveDataset(
          dataset,
          lambda filename: reader(filename, *reader_args),
          cycle_length=reader_num_threads,
          block_length=1,
          sloppy=sloppy_ordering,
          buffer_output_elements=None,
          prefetch_input_elements=None)

    dataset = dataset.apply(apply_fn)

  # Extract values if the `Example` tensors are stored as key-value tuples.
  if dataset_ops.get_legacy_output_types(dataset) == (
      dtypes.string, dtypes.string):
    dataset = dataset_ops.MapDataset(
        dataset, lambda _, v: v, use_inter_op_parallelism=False)

  # Apply dataset repeat and shuffle transformations.
  dataset = _maybe_shuffle_and_repeat(
      dataset, num_epochs, shuffle, shuffle_buffer_size, shuffle_seed)

  # NOTE(mrry): We set `drop_remainder=True` when `num_epochs is None` to
  # improve the shape inference, because it makes the batch dimension static.
  # It is safe to do this because in that case we are repeating the input
  # indefinitely, and all batches will be full-sized.
  dataset = dataset.batch(
      batch_size, drop_remainder=drop_final_batch or num_epochs is None)

  # Parse `Example` tensors to a dictionary of `Feature` tensors.
  dataset = dataset.apply(
      parsing_ops.parse_example_dataset(
          features, num_parallel_calls=parser_num_threads))

  if label_key:
    if label_key not in features:
      raise ValueError(
          "The `label_key` provided (%r) must be one of the `features` keys." %
          label_key)
    dataset = dataset.map(lambda x: (x, x.pop(label_key)))

  dataset = dataset.prefetch(prefetch_buffer_size)
  return dataset


@tf_export(v1=["data.experimental.make_batched_features_dataset"])
def make_batched_features_dataset_v1(file_pattern,  # pylint: disable=missing-docstring
                                     batch_size,
                                     features,
                                     reader=None,
                                     label_key=None,
                                     reader_args=None,
                                     num_epochs=None,
                                     shuffle=True,
                                     shuffle_buffer_size=10000,
                                     shuffle_seed=None,
                                     prefetch_buffer_size=None,
                                     reader_num_threads=None,
                                     parser_num_threads=None,
                                     sloppy_ordering=False,
                                     drop_final_batch=False):
  return dataset_ops.DatasetV1Adapter(make_batched_features_dataset_v2(
      file_pattern, batch_size, features, reader, label_key, reader_args,
      num_epochs, shuffle, shuffle_buffer_size, shuffle_seed,
      prefetch_buffer_size, reader_num_threads, parser_num_threads,
      sloppy_ordering, drop_final_batch))
make_batched_features_dataset_v1.__doc__ = (
    make_batched_features_dataset_v2.__doc__)


def _get_file_names(file_pattern, shuffle):
  """Parse list of file names from pattern, optionally shuffled.

  Args:
    file_pattern: File glob pattern, or list of glob patterns.
    shuffle: Whether to shuffle the order of file names.

  Returns:
    List of file names matching `file_pattern`.

  Raises:
    ValueError: If `file_pattern` is empty, or pattern matches no files.
  """
  if isinstance(file_pattern, list):
    if not file_pattern:
      raise ValueError("File pattern is empty.")
    file_names = []
    for entry in file_pattern:
      file_names.extend(gfile.Glob(entry))
  else:
    file_names = list(gfile.Glob(file_pattern))

  if not file_names:
    raise ValueError("No files match %s." % file_pattern)

  # Sort files so it will be deterministic for unit tests.
  if not shuffle:
    file_names = sorted(file_names)
  return file_names


@tf_export("data.experimental.SqlDataset", v1=[])
class SqlDatasetV2(dataset_ops.DatasetSource):
  """A `Dataset` consisting of the results from a SQL query."""

  def __init__(self, driver_name, data_source_name, query, output_types):
    """Creates a `SqlDataset`.

    `SqlDataset` allows a user to read data from the result set of a SQL query.
    For example:

    ```python
    dataset = tf.data.experimental.SqlDataset("sqlite", "/foo/bar.sqlite3",
                                              "SELECT name, age FROM people",
                                              (tf.string, tf.int32))
    # Prints the rows of the result set of the above query.
    for element in dataset:
      print(element)
    ```

    Args:
      driver_name: A 0-D `tf.string` tensor containing the database type.
        Currently, the only supported value is 'sqlite'.
      data_source_name: A 0-D `tf.string` tensor containing a connection string
        to connect to the database.
      query: A 0-D `tf.string` tensor containing the SQL query to execute.
      output_types: A tuple of `tf.DType` objects representing the types of the
        columns returned by `query`.
    """
    self._driver_name = ops.convert_to_tensor(
        driver_name, dtype=dtypes.string, name="driver_name")
    self._data_source_name = ops.convert_to_tensor(
        data_source_name, dtype=dtypes.string, name="data_source_name")
    self._query = ops.convert_to_tensor(
        query, dtype=dtypes.string, name="query")
    self._element_spec = nest.map_structure(
        lambda dtype: tensor_spec.TensorSpec([], dtype), output_types)
    variant_tensor = gen_experimental_dataset_ops.sql_dataset(
        self._driver_name, self._data_source_name, self._query,
        **self._flat_structure)
    super(SqlDatasetV2, self).__init__(variant_tensor)

  @property
  def element_spec(self):
    return self._element_spec


@tf_export(v1=["data.experimental.SqlDataset"])
class SqlDatasetV1(dataset_ops.DatasetV1Adapter):
  """A `Dataset` consisting of the results from a SQL query."""

  @functools.wraps(SqlDatasetV2.__init__)
  def __init__(self, driver_name, data_source_name, query, output_types):
    wrapped = SqlDatasetV2(driver_name, data_source_name, query, output_types)
    super(SqlDatasetV1, self).__init__(wrapped)


if tf2.enabled():
  CsvDataset = CsvDatasetV2
  SqlDataset = SqlDatasetV2
  make_batched_features_dataset = make_batched_features_dataset_v2
  make_csv_dataset = make_csv_dataset_v2
else:
  CsvDataset = CsvDatasetV1
  SqlDataset = SqlDatasetV1
  make_batched_features_dataset = make_batched_features_dataset_v1
  make_csv_dataset = make_csv_dataset_v1
