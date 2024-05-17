# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Python API for save and loading a dataset."""

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

COMPRESSION_GZIP = "GZIP"
COMPRESSION_SNAPPY = "NONE"
DATASET_SPEC_FILENAME = "dataset_spec.pb"


@tf_export("data.experimental.save", v1=[])
@deprecation.deprecated(None, "Use `tf.data.Dataset.save(...)` instead.")
def save(dataset,
         path,
         compression=None,
         shard_func=None,
         checkpoint_args=None):
  """Saves the content of the given dataset.

  Example usage:

  >>> import tempfile
  >>> path = os.path.join(tempfile.gettempdir(), "saved_data")
  >>> # Save a dataset
  >>> dataset = tf.data.Dataset.range(2)
  >>> tf.data.experimental.save(dataset, path)
  >>> new_dataset = tf.data.experimental.load(path)
  >>> for elem in new_dataset:
  ...   print(elem)
  tf.Tensor(0, shape=(), dtype=int64)
  tf.Tensor(1, shape=(), dtype=int64)

  The saved dataset is saved in multiple file "shards". By default, the dataset
  output is divided to shards in a round-robin fashion but custom sharding can
  be specified via the `shard_func` function. For example, you can save the
  dataset to using a single shard as follows:

  ```python
  dataset = make_dataset()
  def custom_shard_func(element):
    return np.int64(0)
  dataset = tf.data.experimental.save(
      path="/path/to/data", ..., shard_func=custom_shard_func)
  ```

  To enable checkpointing, pass in `checkpoint_args` to the `save` method
  as follows:

  ```python
  dataset = tf.data.Dataset.range(100)
  save_dir = "..."
  checkpoint_prefix = "..."
  step_counter = tf.Variable(0, trainable=False)
  checkpoint_args = {
    "checkpoint_interval": 50,
    "step_counter": step_counter,
    "directory": checkpoint_prefix,
    "max_to_keep": 20,
  }
  dataset.save(dataset, save_dir, checkpoint_args=checkpoint_args)
  ```

  NOTE: The directory layout and file format used for saving the dataset is
  considered an implementation detail and may change. For this reason, datasets
  saved through `tf.data.experimental.save` should only be consumed through
  `tf.data.experimental.load`, which is guaranteed to be backwards compatible.

  Args:
    dataset: The dataset to save.
    path: Required. A directory to use for saving the dataset.
    compression: Optional. The algorithm to use to compress data when writing
      it. Supported options are `GZIP` and `NONE`. Defaults to `NONE`.
    shard_func: Optional. A function to control the mapping of dataset elements
      to file shards. The function is expected to map elements of the input
      dataset to int64 shard IDs. If present, the function will be traced and
      executed as graph computation.
    checkpoint_args: Optional args for checkpointing which will be passed into
      the `tf.train.CheckpointManager`. If `checkpoint_args` are not specified,
      then checkpointing will not be performed. The `save()` implementation
      creates a `tf.train.Checkpoint` object internally, so users should not
      set the `checkpoint` argument in `checkpoint_args`.

  Returns:
    An operation which when executed performs the save. When writing
    checkpoints, returns None. The return value is useful in unit tests.

  Raises:
    ValueError if `checkpoint` is passed into `checkpoint_args`.
  """
  return dataset.save(path, compression, shard_func, checkpoint_args)


@tf_export("data.experimental.load", v1=[])
@deprecation.deprecated(None, "Use `tf.data.Dataset.load(...)` instead.")
def load(path, element_spec=None, compression=None, reader_func=None):
  """Loads a previously saved dataset.

  Example usage:

  >>> import tempfile
  >>> path = os.path.join(tempfile.gettempdir(), "saved_data")
  >>> # Save a dataset
  >>> dataset = tf.data.Dataset.range(2)
  >>> tf.data.experimental.save(dataset, path)
  >>> new_dataset = tf.data.experimental.load(path)
  >>> for elem in new_dataset:
  ...   print(elem)
  tf.Tensor(0, shape=(), dtype=int64)
  tf.Tensor(1, shape=(), dtype=int64)


  If the default option of sharding the saved dataset was used, the element
  order of the saved dataset will be preserved when loading it.

  The `reader_func` argument can be used to specify a custom order in which
  elements should be loaded from the individual shards. The `reader_func` is
  expected to take a single argument -- a dataset of datasets, each containing
  elements of one of the shards -- and return a dataset of elements. For
  example, the order of shards can be shuffled when loading them as follows:

  ```python
  def custom_reader_func(datasets):
    datasets = datasets.shuffle(NUM_SHARDS)
    return datasets.interleave(lambda x: x, num_parallel_calls=AUTOTUNE)

  dataset = tf.data.experimental.load(
      path="/path/to/data", ..., reader_func=custom_reader_func)
  ```

  Args:
    path: Required. A path pointing to a previously saved dataset.
    element_spec: Optional. A nested structure of `tf.TypeSpec` objects matching
      the structure of an element of the saved dataset and specifying the type
      of individual element components. If not provided, the nested structure of
      `tf.TypeSpec` saved with the saved dataset is used. Note that this
      argument is required in graph mode.
    compression: Optional. The algorithm to use to decompress the data when
      reading it. Supported options are `GZIP` and `NONE`. Defaults to `NONE`.
    reader_func: Optional. A function to control how to read data from shards.
      If present, the function will be traced and executed as graph computation.

  Returns:
    A `tf.data.Dataset` instance.

  Raises:
    FileNotFoundError: If `element_spec` is not specified and the saved nested
      structure of `tf.TypeSpec` can not be located with the saved dataset.
    ValueError: If `element_spec` is not specified and the method is executed
      in graph mode.
  """
  return dataset_ops.Dataset.load(path, element_spec, compression, reader_func)
