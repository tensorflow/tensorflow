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
"""Arrow Dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import pyarrow as pa

from tensorflow.contrib.arrow.python.ops import arrow_op_loader  # pylint: disable=unused-import
from tensorflow.contrib.arrow.python.ops import gen_dataset_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape


# TODO expose arrow-cpp to tensor type
def arrow_to_tensor_type(pa_t):
  """Conert Arrow to Tensor dtype."""
  if pa.types.is_int8(pa_t):
    tf_t = dtypes.int8
  elif pa.types.is_int16(pa_t):
    tf_t = dtypes.int16
  elif pa.types.is_int32(pa_t):
    tf_t = dtypes.int32
  elif pa.types.is_int64(pa_t):
    tf_t = dtypes.int64
  elif pa.types.is_uint8(pa_t):
    tf_t = dtypes.uint8
  elif pa.types.is_uint16(pa_t):
    tf_t = dtypes.uint16
  elif pa.types.is_uint32(pa_t):
    tf_t = dtypes.uint32
  elif pa.types.is_uint64(pa_t):
    tf_t = dtypes.uint64
  elif pa.types.is_float16(pa_t):
    tf_t = dtypes.float16
  elif pa.types.is_float32(pa_t):
    tf_t = dtypes.float32
  elif pa.types.is_float64(pa_t):
    tf_t = dtypes.float64
  elif pa.types.is_list(pa_t):
    if pa.types.is_list(pa_t.value_type):
      raise TypeError("Nested arrays are not supported: " + str(pa_t))
    tf_t = arrow_to_tensor_type(pa_t.value_type)
  else:
    raise TypeError("Unsupported type in conversion from Arrow: " + str(pa_t))
  return tf_t


def arrow_schema_to_tensor_types(schema):
  """Convert an Arrow schema to list of Tensor dtypes."""
  return tuple([arrow_to_tensor_type(field.type) for field in schema])


class ArrowBaseDataset(dataset_ops.DatasetSource):

  def __init__(self, columns, output_types):
    self._columns = columns
    self._output_types = output_types

  @property
  def output_classes(self):
    return nest.map_structure(lambda _: ops.Tensor, self._output_types)

  @property
  def output_shapes(self):
    # TODO what about array types?
    return nest.map_structure(lambda _: tensor_shape.TensorShape([]), self._output_types)

  @property
  def output_types(self):
    return self._output_types


class ArrowDataset(ArrowBaseDataset):
  """An Arrow Dataset from record batches in memory, or a Pandas DataFrame.
  """

  def __init__(self,
               record_batches,
               columns,
               output_types):
    """Create an ArrowDataset directly from Arrow record batches.

    Args:
      record_batches: An Arrow record batch or sequence of record batches
    """
    super(ArrowDataset, self).__init__(columns, output_types)
    if isinstance(record_batches, pa.RecordBatch):
      record_batches = [record_batches]
    assert record_batches
    buf = io.BytesIO()
    writer = pa.RecordBatchFileWriter(buf, record_batches[0].schema)
    for batch in record_batches:
      writer.write_batch(batch)
    writer.close()

    self._serialized_batches = ops.convert_to_tensor(
        buf.getvalue(), dtype=dtypes.string, name="serialized_batches")

  def _as_variant_tensor(self):
    return gen_dataset_ops.arrow_dataset(self._serialized_batches,
                                         self._columns,
                                         nest.flatten(self.output_types),
                                         nest.flatten(self.output_shapes))

  @classmethod
  def from_pandas(cls, df, columns=None, preserve_index=True):
    """Create an ArrowDataset from a given Pandas DataFrame.

    Args:
      df: a Pandas DataFrame
      columns: Optional column indices to use, if None all are used
      preserve_index: Flag to include the DataFrame index as a column
    """
    if columns is not None:
      df = df[columns]
    batch = pa.RecordBatch.from_pandas(df, preserve_index=preserve_index)
    columns = tuple(range(len(df.columns)))
    output_types = arrow_schema_to_tensor_types(batch.schema)
    return cls(batch, columns, output_types)


class ArrowFeatherDataset(ArrowBaseDataset):
  """An Arrow Dataset for reading record batches from Arrow feather files.
  Feather is a light-weight columnar format ideal for simple writing of
  Pandas DataFrames. Pyarrow can be used for reading/writing Feather files,
  see https://arrow.apache.org/docs/python/ipc.html#feather-format
  """

  def __init__(self,
               filenames,
               columns,
               output_types):
    """Create an ArrowDataset from one or more Feather file names.

    Args:
      filenames: A `tf.string` tensor, Python list or scalar containing files
      in Arrow Feather format
    """
    super(ArrowFeatherDataset, self).__init__(columns, output_types)
    self._filenames = ops.convert_to_tensor(
        filenames, dtype=dtypes.string, name="filenames")

  def _as_variant_tensor(self):
    return gen_dataset_ops.\
        arrow_feather_dataset(self._filenames,
                              self._columns,
                              nest.flatten(self.output_types),
                              nest.flatten(self.output_shapes))


class ArrowStreamDataset(ArrowBaseDataset):
  """An Arrow Dataset for reading record batches from an input stream.
  Currently supported input streams are a socket client or stdin.
  """

  def __init__(self,
               host,
               columns,
               output_types):
    """Create an ArrowDataset from an input stream.

    Args:
      host: A `tf.string` tensor or Python string defining the input stream
      type. For a socket client, use "<HOST_IP>:<PORT>". For stdin use "STDIN".
    """
    super(ArrowStreamDataset, self).__init__(columns, output_types)
    self._host = ops.convert_to_tensor(
        host, dtype=dtypes.string, name="host")

  def _as_variant_tensor(self):
    return gen_dataset_ops.\
        arrow_stream_dataset(self._host,
                             self._columns,
                             nest.flatten(self.output_types),
                             nest.flatten(self.output_shapes))
