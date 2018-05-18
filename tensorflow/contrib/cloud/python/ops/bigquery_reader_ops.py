# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""BigQuery reading support for TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.cloud.python.ops import gen_bigquery_reader_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import io_ops


class BigQueryReader(io_ops.ReaderBase):
  """A Reader that outputs keys and tf.Example values from a BigQuery table.

  Example use:
    ```python
    # Assume a BigQuery has the following schema,
    #     name      STRING,
    #     age       INT,
    #     state     STRING

    # Create the parse_examples list of features.
    features = dict(
      name=tf.FixedLenFeature([1], tf.string),
      age=tf.FixedLenFeature([1], tf.int32),
      state=tf.FixedLenFeature([1], dtype=tf.string, default_value="UNK"))

    # Create a Reader.
    reader = bigquery_reader_ops.BigQueryReader(project_id=PROJECT,
                                                dataset_id=DATASET,
                                                table_id=TABLE,
                                                timestamp_millis=TIME,
                                                num_partitions=NUM_PARTITIONS,
                                                features=features)

    # Populate a queue with the BigQuery Table partitions.
    queue = tf.train.string_input_producer(reader.partitions())

    # Read and parse examples.
    row_id, examples_serialized = reader.read(queue)
    examples = tf.parse_example(examples_serialized, features=features)

    # Process the Tensors examples["name"], examples["age"], etc...
    ```

  Note that to create a reader a snapshot timestamp is necessary. This
  will enable the reader to look at a consistent snapshot of the table.
  For more information, see 'Table Decorators' in BigQuery docs.

  See ReaderBase for supported methods.
  """

  def __init__(self,
               project_id,
               dataset_id,
               table_id,
               timestamp_millis,
               num_partitions,
               features=None,
               columns=None,
               test_end_point=None,
               name=None):
    """Creates a BigQueryReader.

    Args:
      project_id: GCP project ID.
      dataset_id: BigQuery dataset ID.
      table_id: BigQuery table ID.
      timestamp_millis: timestamp to snapshot the table in milliseconds since
        the epoch. Relative (negative or zero) snapshot times are not allowed.
        For more details, see 'Table Decorators' in BigQuery docs.
      num_partitions: Number of non-overlapping partitions to read from.
      features: parse_example compatible dict from keys to `VarLenFeature` and
        `FixedLenFeature` objects.  Keys are read as columns from the db.
      columns: list of columns to read, can be set iff features is None.
      test_end_point: Used only for testing purposes (optional).
      name: a name for the operation (optional).

    Raises:
      TypeError: - If features is neither None nor a dict or
                 - If columns is neither None nor a list or
                 - If both features and columns are None or set.
    """
    if (features is None) == (columns is None):
      raise TypeError("exactly one of features and columns must be set.")

    if features is not None:
      if not isinstance(features, dict):
        raise TypeError("features must be a dict.")
      self._columns = list(features.keys())
    elif columns is not None:
      if not isinstance(columns, list):
        raise TypeError("columns must be a list.")
      self._columns = columns

    self._project_id = project_id
    self._dataset_id = dataset_id
    self._table_id = table_id
    self._timestamp_millis = timestamp_millis
    self._num_partitions = num_partitions
    self._test_end_point = test_end_point

    reader = gen_bigquery_reader_ops.big_query_reader(
        name=name,
        project_id=self._project_id,
        dataset_id=self._dataset_id,
        table_id=self._table_id,
        timestamp_millis=self._timestamp_millis,
        columns=self._columns,
        test_end_point=self._test_end_point)
    super(BigQueryReader, self).__init__(reader)

  def partitions(self, name=None):
    """Returns serialized BigQueryTablePartition messages.

    These messages represent a non-overlapping division of a table for a
    bulk read.

    Args:
      name: a name for the operation (optional).

    Returns:
      `1-D` string `Tensor` of serialized `BigQueryTablePartition` messages.
    """
    return gen_bigquery_reader_ops.generate_big_query_reader_partitions(
        name=name,
        project_id=self._project_id,
        dataset_id=self._dataset_id,
        table_id=self._table_id,
        timestamp_millis=self._timestamp_millis,
        num_partitions=self._num_partitions,
        test_end_point=self._test_end_point,
        columns=self._columns)


ops.NotDifferentiable("BigQueryReader")
