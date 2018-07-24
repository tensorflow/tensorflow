# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""The Python API for TensorFlow's Bigtable integration.

TensorFlow has support for reading from and writing to Cloud Bigtable. To use
the Bigtable TensorFlow integration, first create a BigtableClient (which
configures your connection to Cloud Bigtable), and then open a Table. The Table
object then allows you to create numerous @{tf.data.Dataset}s to read data, or
write a @{tf.data.Dataset} object to the underlying Bigtable Table.

For background on Google Cloud Bigtable, see: https://cloud.google.com/bigtable.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six import iteritems
from six import string_types

from tensorflow.contrib.bigtable.ops import gen_bigtable_ops
from tensorflow.contrib.data.python.ops import interleave_ops
from tensorflow.contrib.util import loader
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import resource_loader

_bigtable_so = loader.load_op_library(
    resource_loader.get_path_to_datafile("_bigtable.so"))


class BigtableClient(object):
  """BigtableClient is the entrypoint for interacting with Cloud Bigtable in TF.

  BigtableClient encapsulates a connection to Cloud Bigtable, and exposes the
  `table` method to open a Bigtable Table.
  """

  def __init__(self,
               project_id,
               instance_id,
               connection_pool_size=None,
               max_receive_message_size=None):
    """Creates a BigtableClient that can be used to open connections to tables.

    Args:
      project_id: A string representing the GCP project id to connect to.
      instance_id: A string representing the Bigtable instance to connect to.
      connection_pool_size: (Optional.) A number representing the number of
        concurrent connections to the Cloud Bigtable service to make.
      max_receive_message_size: (Optional.) The maximum bytes received in a
        single gRPC response.

    Raises:
      ValueError: if the arguments are invalid (e.g. wrong type, or out of
        expected ranges (e.g. negative).)
    """
    if not isinstance(project_id, str):
      raise ValueError("`project_id` must be a string")
    self._project_id = project_id

    if not isinstance(instance_id, str):
      raise ValueError("`instance_id` must be a string")
    self._instance_id = instance_id

    if connection_pool_size is None:
      connection_pool_size = -1
    elif connection_pool_size < 1:
      raise ValueError("`connection_pool_size` must be positive")

    if max_receive_message_size is None:
      max_receive_message_size = -1
    elif max_receive_message_size < 1:
      raise ValueError("`max_receive_message_size` must be positive")

    self._connection_pool_size = connection_pool_size

    self._resource = gen_bigtable_ops.bigtable_client(
        project_id, instance_id, connection_pool_size, max_receive_message_size)

  def table(self, name, snapshot=None):
    """Opens a table and returns a `BigtableTable` object.

    Args:
      name: A `tf.string` `tf.Tensor` name of the table to open.
      snapshot: Either a `tf.string` `tf.Tensor` snapshot id, or `True` to
        request the creation of a snapshot. (Note: currently unimplemented.)

    Returns:
      A `BigtableTable` python object representing the operations available on
      the table.
    """
    # TODO(saeta): Implement snapshot functionality.
    table = gen_bigtable_ops.bigtable_table(self._resource, name)
    return BigtableTable(name, snapshot, table)


class BigtableTable(object):
  """BigtableTable is the entrypoint for reading and writing data in Cloud
  Bigtable.

  This BigtableTable class is the Python representation of the Cloud Bigtable
  table within TensorFlow. Methods on this class allow data to be read from and
  written to the Cloud Bigtable service in flexible and high performance
  manners.
  """

  # TODO(saeta): Investigate implementing tf.contrib.lookup.LookupInterface.
  # TODO(saeta): Consider variant tensors instead of resources (while supporting
  #    connection pooling).

  def __init__(self, name, snapshot, resource):
    self._name = name
    self._snapshot = snapshot
    self._resource = resource

  def lookup_columns(self, *args, **kwargs):
    """Retrieves the values of columns for a dataset of keys.

    Example usage:
    ```
    table = bigtable_client.table("my_table")
    key_dataset = table.get_keys_prefix("imagenet")
    images = key_dataset.apply(table.lookup_columns(("cf1", "image"),
                                                    ("cf2", "label"),
                                                    ("cf2", "boundingbox")))
    training_data = images.map(parse_and_crop, num_parallel_calls=64).batch(128)
    ```

    Alternatively, you can use keyword arguments to specify the columns to
    capture. Example (same as above, rewritten):
    ```
    table = bigtable_client.table("my_table")
    key_dataset = table.get_keys_prefix("imagenet")
    images = key_dataset.apply(table.lookup_columns(
        cf1="image", cf2=("label", "boundingbox")))
    training_data = images.map(parse_and_crop, num_parallel_calls=64).batch(128)
    ```

    Note: certain kwargs keys are reserved, and thus some column families cannot
    be identified using the kwargs syntax. Instead, please use the args syntax.
    This list includes:
      - 'name'
    This list can change at any time.

    Args:
      *args: A list of tuples containing (column family, column name) pairs.
      **kwargs: Column families and

    Returns:
      A function that can be passed to `tf.data.Dataset.apply` to retrieve the
      values of columns for the rows.
    """
    table = self  # Capture self
    normalized = args
    if normalized is None:
      normalized = []
    if isinstance(normalized, tuple):
      normalized = list(normalized)
    for key, value in iteritems(kwargs):
      if key == "name":
        continue
      if isinstance(value, str):
        normalized.append((key, value))
        continue
      for col in value:
        normalized.append((key, col))

    def _apply_fn(dataset):
      # TODO(saeta): Verify dataset's types are correct!
      return _BigtableLookupDataset(dataset, table, normalized)

    return _apply_fn

  def keys_by_range_dataset(self, start, end):
    """Retrieves all row keys between start and end.

    Note: it does NOT retrieve the values of columns.

    Args:
      start: The start row key. The row keys for rows after start (inclusive)
        will be retrieved.
      end: (Optional.) The end row key. Rows up to (but not including) end will
        be retrieved. If end is None, all subsequent row keys will be retrieved.

    Returns:
      A @{tf.data.Dataset} containing `tf.string` Tensors corresponding to all
      of the row keys between `start` and `end`.
    """
    # TODO(saeta): Make inclusive / exclusive configurable?
    if end is None:
      end = ""
    return _BigtableRangeKeyDataset(self, start, end)

  def keys_by_prefix_dataset(self, prefix):
    """Retrieves the row keys matching a given prefix.

    Args:
      prefix: All row keys that begin with `prefix` in the table will be
        retrieved.

    Returns:
      A @{tf.data.Dataset}. containing `tf.string` Tensors corresponding to all
      of the row keys matching that prefix.
    """
    return _BigtablePrefixKeyDataset(self, prefix)

  def sample_keys(self):
    """Retrieves a sampling of row keys from the Bigtable table.

    This dataset is most often used in conjunction with
    @{tf.contrib.data.parallel_interleave} to construct a set of ranges for
    scanning in parallel.

    Returns:
      A @{tf.data.Dataset} returning string row keys.
    """
    return _BigtableSampleKeysDataset(self)

  def scan_prefix(self, prefix, probability=None, columns=None, **kwargs):
    """Retrieves row (including values) from the Bigtable service.

    Rows with row-key prefixed by `prefix` will be retrieved.

    Specifying the columns to retrieve for each row is done by either using
    kwargs or in the columns parameter. To retrieve values of the columns "c1",
    and "c2" from the column family "cfa", and the value of the column "c3"
    from column family "cfb", the following datasets (`ds1`, and `ds2`) are
    equivalent:

    ```
    table = # ...
    ds1 = table.scan_prefix("row_prefix", columns=[("cfa", "c1"),
                                                   ("cfa", "c2"),
                                                   ("cfb", "c3")])
    ds2 = table.scan_prefix("row_prefix", cfa=["c1", "c2"], cfb="c3")
    ```

    Note: only the latest value of a cell will be retrieved.

    Args:
      prefix: The prefix all row keys must match to be retrieved for prefix-
        based scans.
      probability: (Optional.) A float between 0 (exclusive) and 1 (inclusive).
        A non-1 value indicates to probabilistically sample rows with the
        provided probability.
      columns: The columns to read. Note: most commonly, they are expressed as
        kwargs. Use the columns value if you are using column families that are
        reserved. The value of columns and kwargs are merged. Columns is a list
        of tuples of strings ("column_family", "column_qualifier").
      **kwargs: The column families and columns to read. Keys are treated as
        column_families, and values can be either lists of strings, or strings
        that are treated as the column qualifier (column name).

    Returns:
      A @{tf.data.Dataset} returning the row keys and the cell contents.

    Raises:
      ValueError: If the configured probability is unexpected.
    """
    probability = _normalize_probability(probability)
    normalized = _normalize_columns(columns, kwargs)
    return _BigtableScanDataset(self, prefix, "", "", normalized, probability)

  def scan_range(self, start, end, probability=None, columns=None, **kwargs):
    """Retrieves rows (including values) from the Bigtable service.

    Rows with row-keys between `start` and `end` will be retrieved.

    Specifying the columns to retrieve for each row is done by either using
    kwargs or in the columns parameter. To retrieve values of the columns "c1",
    and "c2" from the column family "cfa", and the value of the column "c3"
    from column family "cfb", the following datasets (`ds1`, and `ds2`) are
    equivalent:

    ```
    table = # ...
    ds1 = table.scan_range("row_start", "row_end", columns=[("cfa", "c1"),
                                                            ("cfa", "c2"),
                                                            ("cfb", "c3")])
    ds2 = table.scan_range("row_start", "row_end", cfa=["c1", "c2"], cfb="c3")
    ```

    Note: only the latest value of a cell will be retrieved.

    Args:
      start: The start of the range when scanning by range.
      end: (Optional.) The end of the range when scanning by range.
      probability: (Optional.) A float between 0 (exclusive) and 1 (inclusive).
        A non-1 value indicates to probabilistically sample rows with the
        provided probability.
      columns: The columns to read. Note: most commonly, they are expressed as
        kwargs. Use the columns value if you are using column families that are
        reserved. The value of columns and kwargs are merged. Columns is a list
        of tuples of strings ("column_family", "column_qualifier").
      **kwargs: The column families and columns to read. Keys are treated as
        column_families, and values can be either lists of strings, or strings
        that are treated as the column qualifier (column name).

    Returns:
      A @{tf.data.Dataset} returning the row keys and the cell contents.

    Raises:
      ValueError: If the configured probability is unexpected.
    """
    probability = _normalize_probability(probability)
    normalized = _normalize_columns(columns, kwargs)
    return _BigtableScanDataset(self, "", start, end, normalized, probability)

  def parallel_scan_prefix(self,
                           prefix,
                           num_parallel_scans=None,
                           probability=None,
                           columns=None,
                           **kwargs):
    """Retrieves row (including values) from the Bigtable service at high speed.

    Rows with row-key prefixed by `prefix` will be retrieved. This method is
    similar to `scan_prefix`, but by constrast performs multiple sub-scans in
    parallel in order to achieve higher performance.

    Note: The dataset produced by this method is not deterministic!

    Specifying the columns to retrieve for each row is done by either using
    kwargs or in the columns parameter. To retrieve values of the columns "c1",
    and "c2" from the column family "cfa", and the value of the column "c3"
    from column family "cfb", the following datasets (`ds1`, and `ds2`) are
    equivalent:

    ```
    table = # ...
    ds1 = table.parallel_scan_prefix("row_prefix", columns=[("cfa", "c1"),
                                                            ("cfa", "c2"),
                                                            ("cfb", "c3")])
    ds2 = table.parallel_scan_prefix("row_prefix", cfa=["c1", "c2"], cfb="c3")
    ```

    Note: only the latest value of a cell will be retrieved.

    Args:
      prefix: The prefix all row keys must match to be retrieved for prefix-
        based scans.
      num_parallel_scans: (Optional.) The number of concurrent scans against the
        Cloud Bigtable instance.
      probability: (Optional.) A float between 0 (exclusive) and 1 (inclusive).
        A non-1 value indicates to probabilistically sample rows with the
        provided probability.
      columns: The columns to read. Note: most commonly, they are expressed as
        kwargs. Use the columns value if you are using column families that are
        reserved. The value of columns and kwargs are merged. Columns is a list
        of tuples of strings ("column_family", "column_qualifier").
      **kwargs: The column families and columns to read. Keys are treated as
        column_families, and values can be either lists of strings, or strings
        that are treated as the column qualifier (column name).

    Returns:
      A @{tf.data.Dataset} returning the row keys and the cell contents.

    Raises:
      ValueError: If the configured probability is unexpected.
    """
    probability = _normalize_probability(probability)
    normalized = _normalize_columns(columns, kwargs)
    ds = _BigtableSampleKeyPairsDataset(self, prefix, "", "")
    return self._make_parallel_scan_dataset(ds, num_parallel_scans, probability,
                                            normalized)

  def parallel_scan_range(self,
                          start,
                          end,
                          num_parallel_scans=None,
                          probability=None,
                          columns=None,
                          **kwargs):
    """Retrieves rows (including values) from the Bigtable service.

    Rows with row-keys between `start` and `end` will be retrieved. This method
    is similar to `scan_range`, but by constrast performs multiple sub-scans in
    parallel in order to achieve higher performance.

    Note: The dataset produced by this method is not deterministic!

    Specifying the columns to retrieve for each row is done by either using
    kwargs or in the columns parameter. To retrieve values of the columns "c1",
    and "c2" from the column family "cfa", and the value of the column "c3"
    from column family "cfb", the following datasets (`ds1`, and `ds2`) are
    equivalent:

    ```
    table = # ...
    ds1 = table.parallel_scan_range("row_start",
                                    "row_end",
                                    columns=[("cfa", "c1"),
                                             ("cfa", "c2"),
                                             ("cfb", "c3")])
    ds2 = table.parallel_scan_range("row_start", "row_end",
                                    cfa=["c1", "c2"], cfb="c3")
    ```

    Note: only the latest value of a cell will be retrieved.

    Args:
      start: The start of the range when scanning by range.
      end: (Optional.) The end of the range when scanning by range.
      num_parallel_scans: (Optional.) The number of concurrent scans against the
        Cloud Bigtable instance.
      probability: (Optional.) A float between 0 (exclusive) and 1 (inclusive).
        A non-1 value indicates to probabilistically sample rows with the
        provided probability.
      columns: The columns to read. Note: most commonly, they are expressed as
        kwargs. Use the columns value if you are using column families that are
        reserved. The value of columns and kwargs are merged. Columns is a list
        of tuples of strings ("column_family", "column_qualifier").
      **kwargs: The column families and columns to read. Keys are treated as
        column_families, and values can be either lists of strings, or strings
        that are treated as the column qualifier (column name).

    Returns:
      A @{tf.data.Dataset} returning the row keys and the cell contents.

    Raises:
      ValueError: If the configured probability is unexpected.
    """
    probability = _normalize_probability(probability)
    normalized = _normalize_columns(columns, kwargs)
    ds = _BigtableSampleKeyPairsDataset(self, "", start, end)
    return self._make_parallel_scan_dataset(ds, num_parallel_scans, probability,
                                            normalized)

  def write(self, dataset, column_families, columns, timestamp=None):
    """Writes a dataset to the table.

    Args:
      dataset: A @{tf.data.Dataset} to be written to this table. It must produce
        a list of number-of-columns+1 elements, all of which must be strings.
        The first value will be used as the row key, and subsequent values will
        be used as cell values for the corresponding columns from the
        corresponding column_families and columns entries.
      column_families: A @{tf.Tensor} of `tf.string`s corresponding to the
        column names to store the dataset's elements into.
      columns: A `tf.Tensor` of `tf.string`s corresponding to the column names
        to store the dataset's elements into.
      timestamp: (Optional.) An int64 timestamp to write all the values at.
        Leave as None to use server-provided timestamps.

    Returns:
      A @{tf.Operation} that can be run to perform the write.

    Raises:
      ValueError: If there are unexpected or incompatible types, or if the
        number of columns and column_families does not match the output of
        `dataset`.
    """
    if timestamp is None:
      timestamp = -1  # Bigtable server provided timestamp.
    for tensor_type in nest.flatten(dataset.output_types):
      if tensor_type != dtypes.string:
        raise ValueError("Not all elements of the dataset were `tf.string`")
    for shape in nest.flatten(dataset.output_shapes):
      if not shape.is_compatible_with(tensor_shape.scalar()):
        raise ValueError("Not all elements of the dataset were scalars")
    if len(column_families) != len(columns):
      raise ValueError("len(column_families) != len(columns)")
    if len(nest.flatten(dataset.output_types)) != len(columns) + 1:
      raise ValueError("A column name must be specified for every component of "
                       "the dataset elements. (e.g.: len(columns) != "
                       "len(dataset.output_types))")
    return gen_bigtable_ops.dataset_to_bigtable(
        self._resource,
        dataset._as_variant_tensor(),  # pylint: disable=protected-access
        column_families,
        columns,
        timestamp)

  def _make_parallel_scan_dataset(self, ds, num_parallel_scans,
                                  normalized_probability, normalized_columns):
    """Builds a parallel dataset from a given range.

    Args:
      ds: A `_BigtableSampleKeyPairsDataset` returning ranges of keys to use.
      num_parallel_scans: The number of concurrent parallel scans to use.
      normalized_probability: A number between 0 and 1 for the keep probability.
      normalized_columns: The column families and column qualifiers to retrieve.

    Returns:
      A @{tf.data.Dataset} representing the result of the parallel scan.
    """
    if num_parallel_scans is None:
      num_parallel_scans = 50

    ds = ds.shuffle(buffer_size=10000)  # TODO(saeta): Make configurable.

    def _interleave_fn(start, end):
      return _BigtableScanDataset(
          self,
          prefix="",
          start=start,
          end=end,
          normalized=normalized_columns,
          probability=normalized_probability)

    # Note prefetch_input_elements must be set in order to avoid rpc timeouts.
    ds = ds.apply(
        interleave_ops.parallel_interleave(
            _interleave_fn,
            cycle_length=num_parallel_scans,
            sloppy=True,
            prefetch_input_elements=1))
    return ds


def _normalize_probability(probability):
  if probability is None:
    probability = 1.0
  if isinstance(probability, float) and (probability <= 0.0 or
                                         probability > 1.0):
    raise ValueError("probability must be in the range (0, 1].")
  return probability


def _normalize_columns(columns, provided_kwargs):
  """Converts arguments (columns, and kwargs dict) to C++ representation.

  Args:
    columns: a datastructure containing the column families and qualifier to
      retrieve. Valid types include (1) None, (2) list of tuples, (3) a tuple of
      strings.
    provided_kwargs: a dictionary containing the column families and qualifiers
      to retrieve

  Returns:
    A list of pairs of column family+qualifier to retrieve.

  Raises:
    ValueError: If there are no cells to retrieve or the columns are in an
      incorrect format.
  """
  normalized = columns
  if normalized is None:
    normalized = []
  if isinstance(normalized, tuple):
    if len(normalized) == 2:
      normalized = [normalized]
    else:
      raise ValueError("columns was a tuple of inappropriate length")
  for key, value in iteritems(provided_kwargs):
    if key == "name":
      continue
    if isinstance(value, string_types):
      normalized.append((key, value))
      continue
    for col in value:
      normalized.append((key, col))
  if not normalized:
    raise ValueError("At least one column + column family must be specified.")
  return normalized


class _BigtableKeyDataset(dataset_ops.Dataset):
  """_BigtableKeyDataset is an abstract class representing the keys of a table.
  """

  def __init__(self, table):
    """Constructs a _BigtableKeyDataset.

    Args:
      table: a Bigtable class.
    """
    super(_BigtableKeyDataset, self).__init__()
    self._table = table

  @property
  def output_classes(self):
    return ops.Tensor

  @property
  def output_shapes(self):
    return tensor_shape.TensorShape([])

  @property
  def output_types(self):
    return dtypes.string


class _BigtablePrefixKeyDataset(_BigtableKeyDataset):
  """_BigtablePrefixKeyDataset represents looking up keys by prefix.
  """

  def __init__(self, table, prefix):
    super(_BigtablePrefixKeyDataset, self).__init__(table)
    self._prefix = prefix

  def _as_variant_tensor(self):
    return gen_bigtable_ops.bigtable_prefix_key_dataset(
        table=self._table._resource,  # pylint: disable=protected-access
        prefix=self._prefix)


class _BigtableRangeKeyDataset(_BigtableKeyDataset):
  """_BigtableRangeKeyDataset represents looking up keys by range.
  """

  def __init__(self, table, start, end):
    super(_BigtableRangeKeyDataset, self).__init__(table)
    self._start = start
    self._end = end

  def _as_variant_tensor(self):
    return gen_bigtable_ops.bigtable_range_key_dataset(
        table=self._table._resource,  # pylint: disable=protected-access
        start_key=self._start,
        end_key=self._end)


class _BigtableSampleKeysDataset(_BigtableKeyDataset):
  """_BigtableSampleKeysDataset represents a sampling of row keys.
  """

  # TODO(saeta): Expose the data size offsets into the keys.

  def __init__(self, table):
    super(_BigtableSampleKeysDataset, self).__init__(table)

  def _as_variant_tensor(self):
    return gen_bigtable_ops.bigtable_sample_keys_dataset(
        table=self._table._resource)  # pylint: disable=protected-access


class _BigtableLookupDataset(dataset_ops.Dataset):
  """_BigtableLookupDataset represents a dataset that retrieves values for keys.
  """

  def __init__(self, dataset, table, normalized):
    self._num_outputs = len(normalized) + 1  # 1 for row key
    self._dataset = dataset
    self._table = table
    self._normalized = normalized
    self._column_families = [i[0] for i in normalized]
    self._columns = [i[1] for i in normalized]

  @property
  def output_classes(self):
    return tuple([ops.Tensor] * self._num_outputs)

  @property
  def output_shapes(self):
    return tuple([tensor_shape.TensorShape([])] * self._num_outputs)

  @property
  def output_types(self):
    return tuple([dtypes.string] * self._num_outputs)

  def _as_variant_tensor(self):
    # pylint: disable=protected-access
    return gen_bigtable_ops.bigtable_lookup_dataset(
        keys_dataset=self._dataset._as_variant_tensor(),
        table=self._table._resource,
        column_families=self._column_families,
        columns=self._columns)


class _BigtableScanDataset(dataset_ops.Dataset):
  """_BigtableScanDataset represents a dataset that retrieves keys and values.
  """

  def __init__(self, table, prefix, start, end, normalized, probability):
    self._table = table
    self._prefix = prefix
    self._start = start
    self._end = end
    self._column_families = [i[0] for i in normalized]
    self._columns = [i[1] for i in normalized]
    self._probability = probability
    self._num_outputs = len(normalized) + 1  # 1 for row key

  @property
  def output_classes(self):
    return tuple([ops.Tensor] * self._num_outputs)

  @property
  def output_shapes(self):
    return tuple([tensor_shape.TensorShape([])] * self._num_outputs)

  @property
  def output_types(self):
    return tuple([dtypes.string] * self._num_outputs)

  def _as_variant_tensor(self):
    return gen_bigtable_ops.bigtable_scan_dataset(
        table=self._table._resource,  # pylint: disable=protected-access
        prefix=self._prefix,
        start_key=self._start,
        end_key=self._end,
        column_families=self._column_families,
        columns=self._columns,
        probability=self._probability)


class _BigtableSampleKeyPairsDataset(dataset_ops.Dataset):
  """_BigtableKeyRangeDataset returns key pairs from the Bigtable.
  """

  def __init__(self, table, prefix, start, end):
    self._table = table
    self._prefix = prefix
    self._start = start
    self._end = end

  @property
  def output_classes(self):
    return (ops.Tensor, ops.Tensor)

  @property
  def output_shapes(self):
    return (tensor_shape.TensorShape([]), tensor_shape.TensorShape([]))

  @property
  def output_types(self):
    return (dtypes.string, dtypes.string)

  def _as_variant_tensor(self):
    # pylint: disable=protected-access
    return gen_bigtable_ops.bigtable_sample_key_pairs_dataset(
        table=self._table._resource,
        prefix=self._prefix,
        start_key=self._start,
        end_key=self._end)
