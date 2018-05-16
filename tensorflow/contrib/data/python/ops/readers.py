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

import csv
from math import ceil

import numpy as np

from tensorflow.contrib.data.python.ops import batching
from tensorflow.contrib.data.python.ops import gen_dataset_ops as contrib_gen_dataset_ops
from tensorflow.contrib.data.python.ops import interleave_ops
from tensorflow.contrib.data.python.ops import shuffle_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers as core_readers
from tensorflow.python.data.util import convert
from tensorflow.python.data.util import nest
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import deprecation

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


def _infer_type(str_val, na_value, prev_type, float_dtype):
  """Given a string, infers its tensor type.

  Infers the type of a value by picking the least 'permissive' type possible,
  while still allowing the previous type inference for this column to be valid.

  Args:
    str_val: String value to infer the type of.
    na_value: Additional string to recognize as a NA/NaN CSV value.
    prev_type: Type previously inferred based on values of this column that
      we've seen up till now.
    float_dtype: Either `tf.float32` or `tf.float64`. Denotes what float type
      to parse float strings as.
  Returns:
    Inferred dtype.
  """
  if str_val in ("", na_value):
    return prev_type

  if _is_valid_int32(str_val) and prev_type in (None, dtypes.int32):
    return dtypes.int32

  if _is_valid_int64(str_val) and prev_type in (None, dtypes.int32,
                                                dtypes.int64):
    return dtypes.int64

  if _is_valid_float(str_val, float_dtype) and prev_type != dtypes.string:
    return float_dtype

  return dtypes.string


def _next_csv_row(filenames, num_cols, field_delim, use_quote_delim, header,
                  comment):
  """Generator that yields rows of CSV file(s) in order."""
  for fn in filenames:
    with file_io.FileIO(fn, "r") as f:
      rdr = csv.reader(
          f,
          delimiter=field_delim,
          quoting=csv.QUOTE_MINIMAL if use_quote_delim else csv.QUOTE_NONE)
      if header:
        next(rdr)  # Skip header lines

      for csv_row in rdr:
        if comment is not None and csv_row[0].startswith(comment):
          continue  # Skip comment lines

        if len(csv_row) != num_cols:
          raise ValueError(
              "Problem inferring types: CSV row has different number of fields "
              "than expected.")
        yield csv_row


def _infer_column_defaults(filenames, num_cols, field_delim, use_quote_delim,
                           na_value, header, comment, float_dtype,
                           num_rows_for_inference, select_columns):
  """Infers column types from the first N valid CSV records of files."""
  if select_columns is None:
    select_columns = range(num_cols)
  inferred_types = [None] * len(select_columns)

  for i, csv_row in enumerate(
      _next_csv_row(filenames, num_cols, field_delim, use_quote_delim, header,
                    comment)):
    if num_rows_for_inference is not None and i >= num_rows_for_inference:
      break

    for j, col_index in enumerate(select_columns):
      inferred_types[j] = _infer_type(csv_row[col_index], na_value,
                                      inferred_types[j], float_dtype)

  # Replace None's with a default type
  inferred_types = [t or dtypes.string for t in inferred_types]
  # Default to 0 or '' for null values
  return [
      constant_op.constant([0 if t is not dtypes.string else ""], dtype=t)
      for t in inferred_types
  ]


def _infer_column_names(filenames, field_delim, use_quote_delim):
  """Infers column names from first rows of files."""
  csv_kwargs = {
      "delimiter": field_delim,
      "quoting": csv.QUOTE_MINIMAL if use_quote_delim else csv.QUOTE_NONE
  }
  with file_io.FileIO(filenames[0], "r") as f:
    try:
      column_names = next(csv.reader(f, **csv_kwargs))
    except StopIteration:
      raise ValueError(("Received StopIteration when reading the header line "
                        "of %s.  Empty file?") % filenames[0])

  for name in filenames[1:]:
    with file_io.FileIO(name, "r") as f:
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
  for i, v in enumerate(select_columns):
    if isinstance(v, int):
      if v < 0 or v >= num_cols:
        raise ValueError(
            "Column index %d specified in select_columns out of valid range." %
            v)
      continue
    if v not in names_to_indices:
      raise ValueError(
          "Value '%s' specified in select_columns not a valid column index or "
          "name." % v)
    select_columns[i] = names_to_indices[v]

  # Sort and ensure there are no duplicates
  result = sorted(set(select_columns))
  if len(result) != len(select_columns):
    raise ValueError("select_columns contains duplicate columns")
  return result


def make_csv_dataset(
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
    comment=None,
    num_epochs=None,
    shuffle=True,
    shuffle_buffer_size=10000,
    shuffle_seed=None,
    prefetch_buffer_size=1,
    num_parallel_reads=1,
    num_parallel_parser_calls=2,
    sloppy=False,
    default_float_type=dtypes.float32,
    num_rows_for_inference=100,
):
  """Reads CSV files into a dataset.

  Reads CSV files into a dataset, where each element is a (features, labels)
  tuple that corresponds to a batch of CSV rows. The features dictionary
  maps feature column names to `Tensor`s containing the corresponding
  feature data, and labels is a `Tensor` containing the batch's label data.

  Args:
    file_pattern: List of files or patterns of file paths containing CSV
      records. See @{tf.gfile.Glob} for pattern rules.
    batch_size: An int representing the number of consecutive elements of this
      dataset to combine in a single batch.
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
    comment: An optional character string that marks lines that should not be
      parsed as csv records. If this is provided, all lines that start with
      this character will not be parsed.
    num_epochs: An int specifying the number of times this dataset is repeated.
      If None, cycles through the dataset forever.
    shuffle: A bool that indicates whether the input should be shuffled.
    shuffle_buffer_size: Buffer size to use for shuffling. A large buffer size
      ensures better shuffling, but would increase memory usage and startup
      time.
    shuffle_seed: Randomization seed to use for shuffling.
    prefetch_buffer_size: An int specifying the number of feature batches to
      prefetch for performance improvement. Recommended value is the number of
      batches consumed per training step.
    num_parallel_reads: Number of threads used to read CSV records from files.
      If >1, the results will be interleaved.
    num_parallel_parser_calls: Number of parallel invocations of the CSV parsing
      function on CSV records.
    sloppy: If `True`, reading performance will be improved at
      the cost of non-deterministic ordering. If `False`, the order of elements
      produced is deterministic prior to shuffling (elements are still
      randomized if `shuffle=True`. Note that if the seed is set, then order
      of elements after shuffling is deterministic). Defaults to `False`.
    default_float_type: Either `tf.float32` or `tf.float64`. If defaults are
      not provided, float-like strings are interpreted to be this type.
    num_rows_for_inference: Number of rows of a file to use for type inference
      if record_defaults is not provided. If None, reads all the rows of all
      the files. Defaults to 100.

  Returns:
    A dataset, where each element is a (features, labels) tuple that corresponds
    to a batch of `batch_size` CSV rows. The features dictionary maps feature
    column names to `Tensor`s containing the corresponding column data, and
    labels is a `Tensor` containing the column data for the label column
    specified by `label_name`.

  Raises:
    ValueError: If any of the arguments is malformed.
  """
  # Create dataset of all matching filenames
  filenames = _get_file_names(file_pattern, False)
  dataset = dataset_ops.Dataset.from_tensor_slices(filenames)
  if shuffle:
    dataset = dataset.shuffle(len(filenames), shuffle_seed)

  # Clean arguments; figure out column names and defaults
  if comment is not None and len(comment) != 1:
    raise ValueError("`comment` arg must be a single-character string or None")

  if column_names is None:
    if not header:
      raise ValueError("Cannot infer column names without a header line.")
    # If column names are not provided, infer from the header lines
    column_names = _infer_column_names(filenames, field_delim, use_quote_delim)
  if len(column_names) != len(set(column_names)):
    raise ValueError("Cannot have duplicate column names.")

  if select_columns is not None:
    select_columns = _get_sorted_col_indices(select_columns, column_names)

  if column_defaults is not None:
    column_defaults = [
        constant_op.constant([], dtype=x) if x in _ACCEPTABLE_CSV_TYPES else x
        for x in column_defaults
    ]
  else:
    # If column defaults are not provided, infer from records at graph
    # construction time
    column_defaults = _infer_column_defaults(
        filenames, len(column_names), field_delim, use_quote_delim, na_value,
        header, comment, default_float_type, num_rows_for_inference,
        select_columns)

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

  # Define map and filter functions
  def filter_fn(line):
    return math_ops.not_equal(string_ops.substr(line, 0, 1), comment)

  def filename_to_dataset(filename):
    ds = core_readers.TextLineDataset(filename)
    if header:
      ds = ds.skip(1)
    if comment is not None:
      ds = ds.filter(filter_fn)
    return ds

  def decode_csv(line):
    """Decodes CSV line into features.

    Args:
      line: String tensor corresponding to one csv record.
    Returns:
      A dictionary of feature names to values for that particular record. If
      label_name is provided, extracts the label feature to be returned as the
      second element of the tuple.
    """
    columns = parsing_ops.decode_csv(
        line,
        column_defaults,
        field_delim=field_delim,
        use_quote_delim=use_quote_delim,
        na_value=na_value,
        select_cols=select_columns,
    )
    features = dict(zip(column_names, columns))
    if label_name is not None:
      label = features.pop(label_name)
      return features, label
    return features

  # Read files sequentially or in parallel
  dataset = dataset.apply(
      interleave_ops.parallel_interleave(
          filename_to_dataset, cycle_length=num_parallel_reads, sloppy=sloppy))

  if num_epochs != 1 and shuffle:
    # Use shuffle_and_repeat for perf
    dataset = dataset.apply(
        shuffle_ops.shuffle_and_repeat(shuffle_buffer_size, num_epochs,
                                       shuffle_seed))
  elif shuffle:
    dataset = dataset.shuffle(shuffle_buffer_size, shuffle_seed)
  elif num_epochs != 1:
    dataset = dataset.repeat(num_epochs)

  # Use map_and_batch for perf
  # TODO(b/76425672): use num_parallel_calls for better performance tuning when
  # that is added
  dataset = dataset.apply(
      batching.map_and_batch(
          map_func=decode_csv,
          batch_size=batch_size,
          num_parallel_batches=int(
              ceil(num_parallel_parser_calls / batch_size))))

  dataset = dataset.prefetch(prefetch_buffer_size)
  return dataset


_DEFAULT_READER_BUFFER_SIZE_BYTES = 4 * 1024 * 1024  # 4 MB


class CsvDataset(dataset_ops.Dataset):
  """A Dataset comprising lines from one or more CSV files."""

  def __init__(self,
               filenames,
               record_defaults,
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
    dataset = tf.contrib.data.CsvDataset(
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
    next = dataset.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
      while True:
        try:
          print(sess.run(nxt))
        except tf.errors.OutOfRangeError:
          break

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
    super(CsvDataset, self).__init__()
    self._filenames = ops.convert_to_tensor(
        filenames, dtype=dtypes.string, name="filenames")
    record_defaults = [
        constant_op.constant([], dtype=x) if x in _ACCEPTABLE_CSV_TYPES else x
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
    self._output_shapes = tuple(
        tensor_shape.scalar() for _ in range(len(record_defaults)))
    self._output_types = tuple(d.dtype for d in self._record_defaults)
    self._output_classes = tuple(
        ops.Tensor for _ in range(len(record_defaults)))

  def _as_variant_tensor(self):
    # Constructs graph node for the dataset op.
    return contrib_gen_dataset_ops.csv_dataset(
        filenames=self._filenames,
        record_defaults=self._record_defaults,
        buffer_size=self._buffer_size,
        header=self._header,
        output_shapes=self._output_shapes,
        field_delim=self._field_delim,
        use_quote_delim=self._use_quote_delim,
        na_value=self._na_value,
        select_cols=self._select_cols,
    )

  @property
  def output_types(self):
    return self._output_types

  @property
  def output_shapes(self):
    return self._output_shapes

  @property
  def output_classes(self):
    return self._output_classes


def make_batched_features_dataset(file_pattern,
                                  batch_size,
                                  features,
                                  reader=core_readers.TFRecordDataset,
                                  reader_args=None,
                                  num_epochs=None,
                                  shuffle=True,
                                  shuffle_buffer_size=10000,
                                  shuffle_seed=None,
                                  prefetch_buffer_size=1,
                                  reader_num_threads=1,
                                  parser_num_threads=2,
                                  sloppy_ordering=False,
                                  drop_final_batch=False):
  """Returns a `Dataset` of feature dictionaries from `Example` protos.

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
      `Example` records. See `tf.gfile.Glob` for pattern rules.
    batch_size: An int representing the number of consecutive elements of this
      dataset to combine in a single batch.
    features: A `dict` mapping feature keys to `FixedLenFeature` or
      `VarLenFeature` values. See `tf.parse_example`.
    reader: A function or class that can be
      called with a `filenames` tensor and (optional) `reader_args` and returns
      a `Dataset` of `Example` tensors. Defaults to `tf.data.TFRecordDataset`.
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
      per training step (default is 1).
    reader_num_threads: Number of threads used to read `Example` records. If >1,
      the results will be interleaved.
    parser_num_threads: Number of threads to use for parsing `Example` tensors
      into a dictionary of `Feature` tensors.
    sloppy_ordering: If `True`, reading performance will be improved at
      the cost of non-deterministic ordering. If `False`, the order of elements
      produced is deterministic prior to shuffling (elements are still
      randomized if `shuffle=True`. Note that if the seed is set, then order
      of elements after shuffling is deterministic). Defaults to `False`.
    drop_final_batch: If `True`, and the batch size does not evenly divide the
      input dataset size, the final smaller batch will be dropped. Defaults to
      `False`.

  Returns:
    A dataset of `dict` elements. Each `dict` maps feature keys to
    `Tensor` or `SparseTensor` objects.
  """
  # Create dataset of all matching filenames
  filenames = _get_file_names(file_pattern, False)
  dataset = dataset_ops.Dataset.from_tensor_slices(filenames)
  if shuffle:
    dataset = dataset.shuffle(len(filenames), shuffle_seed)

  # Read `Example` records from files as tensor objects.
  if reader_args is None:
    reader_args = []

  # Read files sequentially (if reader_num_threads=1) or in parallel
  dataset = dataset.apply(
      interleave_ops.parallel_interleave(
          lambda filename: reader(filename, *reader_args),
          cycle_length=reader_num_threads,
          sloppy=sloppy_ordering))

  # Extract values if the `Example` tensors are stored as key-value tuples.
  if dataset.output_types == (dtypes.string, dtypes.string):
    dataset = dataset.map(lambda _, v: v)

  # Apply dataset repeat and shuffle transformations.
  repeat_dataset = (num_epochs != 1)
  if repeat_dataset and shuffle:
    # Used fused shuffle_and_repeat operation for better performance
    dataset = dataset.apply(
        shuffle_ops.shuffle_and_repeat(shuffle_buffer_size, num_epochs,
                                       shuffle_seed))
  elif repeat_dataset:
    dataset = dataset.repeat(num_epochs)
  elif shuffle:
    dataset = dataset.shuffle(shuffle_buffer_size, shuffle_seed)

  if drop_final_batch:
    dataset = dataset.apply(batching.batch_and_drop_remainder(batch_size))
  else:
    dataset = dataset.batch(batch_size)

  # Parse `Example` tensors to a dictionary of `Feature` tensors.
  dataset = dataset.map(
      lambda x: parsing_ops.parse_example(x, features),
      num_parallel_calls=parser_num_threads)

  # TODO(rachelim): Add an optional label_name argument for extracting the label
  # from the features dictionary, to comply with the type expected by the
  # input_fn to a `tf.Estimator.train` or `tf.Estimator.evaluate` function.
  dataset = dataset.prefetch(prefetch_buffer_size)
  return dataset


@deprecation.deprecated(None,
                        "Use `tf.contrib.data.make_batched_features_dataset`")
def read_batch_features(file_pattern,
                        batch_size,
                        features,
                        reader=core_readers.TFRecordDataset,
                        reader_args=None,
                        randomize_input=True,
                        num_epochs=None,
                        capacity=10000):
  """Reads batches of Examples.

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
      `Example` records. See `tf.gfile.Glob` for pattern rules.
    batch_size: An int representing the number of consecutive elements of this
      dataset to combine in a single batch.
    features: A `dict` mapping feature keys to `FixedLenFeature` or
      `VarLenFeature` values. See `tf.parse_example`.
    reader: A function or class that can be
      called with a `filenames` tensor and (optional) `reader_args` and returns
      a `Dataset` of `Example` tensors. Defaults to `tf.data.TFRecordDataset`.
    reader_args: Additional arguments to pass to the reader class.
    randomize_input: Whether the input should be randomized.
    num_epochs: Integer specifying the number of times to read through the
      dataset. If None, cycles through the dataset forever.
    capacity: Buffer size of the ShuffleDataset. A large capacity ensures better
      shuffling but would increase memory usage and startup time.
  Returns:
    A dict from keys in features to `Tensor` or `SparseTensor` objects.
  """
  dataset = make_batched_features_dataset(
      file_pattern,
      batch_size,
      features,
      reader=reader,
      reader_args=reader_args,
      shuffle=randomize_input,
      num_epochs=num_epochs,
      shuffle_buffer_size=capacity)
  iterator = dataset.make_one_shot_iterator()
  outputs = iterator.get_next()
  return outputs


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


class SqlDataset(dataset_ops.Dataset):
  """A `Dataset` consisting of the results from a SQL query."""

  def __init__(self, driver_name, data_source_name, query, output_types):
    """Creates a `SqlDataset`.

    `SqlDataset` allows a user to read data from the result set of a SQL query.
    For example:

    ```python
    dataset = tf.contrib.data.SqlDataset("sqlite", "/foo/bar.sqlite3",
                                         "SELECT name, age FROM people",
                                         (tf.string, tf.int32))
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    # Prints the rows of the result set of the above query.
    while True:
      try:
        print(sess.run(next_element))
      except tf.errors.OutOfRangeError:
        break
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
    super(SqlDataset, self).__init__()
    self._driver_name = ops.convert_to_tensor(
        driver_name, dtype=dtypes.string, name="driver_name")
    self._data_source_name = ops.convert_to_tensor(
        data_source_name, dtype=dtypes.string, name="data_source_name")
    self._query = ops.convert_to_tensor(
        query, dtype=dtypes.string, name="query")
    self._output_types = output_types

  def _as_variant_tensor(self):
    return gen_dataset_ops.sql_dataset(self._driver_name,
                                       self._data_source_name, self._query,
                                       nest.flatten(self.output_types),
                                       nest.flatten(self.output_shapes))

  @property
  def output_classes(self):
    return nest.map_structure(lambda _: ops.Tensor, self._output_types)

  @property
  def output_shapes(self):
    return nest.map_structure(lambda _: tensor_shape.TensorShape([]),
                              self._output_types)

  @property
  def output_types(self):
    return self._output_types
