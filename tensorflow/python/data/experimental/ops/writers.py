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
"""Python wrappers for tf.data writers."""
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import convert
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.types import data as data_types
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


@tf_export("data.experimental.TFRecordWriter")
@deprecation.deprecated(
    None, "To write TFRecords to disk, use `tf.io.TFRecordWriter`. To save "
    "and load the contents of a dataset, use `tf.data.experimental.save` "
    "and `tf.data.experimental.load`")
class TFRecordWriter:
  """Writes a dataset to a TFRecord file.

  The elements of the dataset must be scalar strings. To serialize dataset
  elements as strings, you can use the `tf.io.serialize_tensor` function.

  ```python
  dataset = tf.data.Dataset.range(3)
  dataset = dataset.map(tf.io.serialize_tensor)
  writer = tf.data.experimental.TFRecordWriter("/path/to/file.tfrecord")
  writer.write(dataset)
  ```

  To read back the elements, use `TFRecordDataset`.

  ```python
  dataset = tf.data.TFRecordDataset("/path/to/file.tfrecord")
  dataset = dataset.map(lambda x: tf.io.parse_tensor(x, tf.int64))
  ```

  To shard a `dataset` across multiple TFRecord files:

  ```python
  dataset = ... # dataset to be written

  def reduce_func(key, dataset):
    filename = tf.strings.join([PATH_PREFIX, tf.strings.as_string(key)])
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(dataset.map(lambda _, x: x))
    return tf.data.Dataset.from_tensors(filename)

  dataset = dataset.enumerate()
  dataset = dataset.apply(tf.data.experimental.group_by_window(
    lambda i, _: i % NUM_SHARDS, reduce_func, tf.int64.max
  ))

  # Iterate through the dataset to trigger data writing.
  for _ in dataset:
    pass
  ```
  """

  def __init__(self, filename, compression_type=None):
    """Initializes a `TFRecordWriter`.

    Args:
      filename: a string path indicating where to write the TFRecord data.
      compression_type: (Optional.) a string indicating what type of compression
        to use when writing the file. See `tf.io.TFRecordCompressionType` for
        what types of compression are available. Defaults to `None`.
    """
    self._filename = ops.convert_to_tensor(
        filename, dtypes.string, name="filename")
    self._compression_type = convert.optional_param_to_tensor(
        "compression_type",
        compression_type,
        argument_default="",
        argument_dtype=dtypes.string)

  def write(self, dataset):
    """Writes a dataset to a TFRecord file.

    An operation that writes the content of the specified dataset to the file
    specified in the constructor.

    If the file exists, it will be overwritten.

    Args:
      dataset: a `tf.data.Dataset` whose elements are to be written to a file

    Returns:
      In graph mode, this returns an operation which when executed performs the
      write. In eager mode, the write is performed by the method itself and
      there is no return value.

    Raises
      TypeError: if `dataset` is not a `tf.data.Dataset`.
      TypeError: if the elements produced by the dataset are not scalar strings.
    """
    if not isinstance(dataset, data_types.DatasetV2):
      raise TypeError(
          f"Invalid `dataset.` Expected a `tf.data.Dataset` object but got "
          f"{type(dataset)}."
      )
    if not dataset_ops.get_structure(dataset).is_compatible_with(
        tensor_spec.TensorSpec([], dtypes.string)):
      raise TypeError(
          f"Invalid `dataset`. Expected a`dataset` that produces scalar "
          f"`tf.string` elements, but got a dataset which produces elements "
          f"with shapes {dataset_ops.get_legacy_output_shapes(dataset)} and "
          f"types {dataset_ops.get_legacy_output_types(dataset)}.")
    # pylint: disable=protected-access
    dataset = dataset._apply_debug_options()
    return gen_experimental_dataset_ops.dataset_to_tf_record(
        dataset._variant_tensor, self._filename, self._compression_type)
