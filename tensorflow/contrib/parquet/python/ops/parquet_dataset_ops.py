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
"""Parquet Dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.parquet.python.ops import parquet_op_loader  # pylint: disable=unused-import
from tensorflow.contrib.parquet.python.ops import gen_dataset_ops
from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape


class ParquetDataset(Dataset):
  """A Parquet Dataset that reads the parquet file."""

  def __init__(self, filenames, columns, output_types):
    """Create a `ParquetDataset`.

    `ParquetDataset` allows a user to read data from a parquet file.
    For example:

    ```python
    dataset = tf.contrib.parquet.ParquetDataset(
        "/foo/bar.parquet", [0, 1], (tf.bool, tf.int32))
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    # Prints the rows of the result set of the column [0, 1].
    while True:
      try:
        print(sess.run(next_element))
      except tf.errors.OutOfRangeError:
        break
    ```

    Args:
      filenames: A 0-D or 1-D `tf.string` tensor containing one or more
        filenames.
      columns: A 0-D or 1-D `tf.int32` tensor containing the columns to extract.
      output_types: A tuple of `tf.DType` objects representing the types of the
        columns returned.
    """
    super(ParquetDataset, self).__init__()
    self._filenames = ops.convert_to_tensor(
        filenames, dtype=dtypes.string, name="filenames")
    self._columns = ops.convert_to_tensor(
        columns, dtype=dtypes.int32, name="columns")
    self._output_types = output_types

  def _as_variant_tensor(self):
    return gen_dataset_ops.parquet_dataset(self._filenames, self._columns,
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
