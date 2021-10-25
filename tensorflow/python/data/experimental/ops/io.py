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

import multiprocessing
import os

from tensorflow.python.compat import compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.platform import gfile
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import tracking
from tensorflow.python.util import lazy_loader
from tensorflow.python.util.tf_export import tf_export

COMPRESSION_GZIP = "GZIP"
COMPRESSION_SNAPPY = "NONE"
DATASET_SPEC_FILENAME = "dataset_spec.pb"
# TODO(b/176933539): Use the regular import.
nested_structure_coder = lazy_loader.LazyLoader(
    "nested_structure_coder", globals(),
    "tensorflow.python.saved_model.nested_structure_coder")


@tf_export("data.experimental.save", v1=[])
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
    return 0
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
  Raises:
    ValueError if `checkpoint` is passed into `checkpoint_args`.
  """
  if (context.executing_eagerly() and checkpoint_args
      and compat.forward_compatible(2021, 6, 29)):
    save_dataset = _SaveDataset(dataset, path, shard_func, compression)
    save_iterator = iter(save_dataset)

    if "checkpoint" in checkpoint_args:
      raise ValueError(
          "'Invalid `checkpoint_args`. `checkpoint_args` are not allowed "
          "to include 'checkpoint'."
      )
    checkpoint = tracking.util.Checkpoint(iterator=save_iterator)
    checkpoint_args["checkpoint"] = checkpoint
    manager = checkpoint_management.CheckpointManager(**checkpoint_args)
    checkpoint.restore(manager.latest_checkpoint)

    for _ in enumerate(save_iterator):
      if "step_counter" in checkpoint_args:
        checkpoint_args["step_counter"].assign_add(delta=1)
      manager.save(check_interval=True)
  else:
    dataset, shard_func, use_shard_func, path = _set_save_dataset_attributes(
        dataset, shard_func, path)
    gen_experimental_dataset_ops.save_dataset(
        dataset._variant_tensor,   # pylint: disable=protected-access
        path=path,
        shard_func_other_args=shard_func.captured_inputs,
        compression=compression,
        shard_func=shard_func,
        use_shard_func=use_shard_func)


class _SaveDataset(dataset_ops.UnaryUnchangedStructureDataset):
  """"A dataset that loads previously saved dataset."""

  def __init__(self, dataset, path, shard_func, compression):
    dataset, shard_func, use_shard_func, path = _set_save_dataset_attributes(
        dataset, shard_func, path)
    variant_tensor = gen_experimental_dataset_ops.save_dataset_v2(
        dataset._variant_tensor,   # pylint: disable=protected-access
        path=path,
        shard_func_other_args=shard_func.captured_inputs,
        shard_func=shard_func,
        use_shard_func=use_shard_func,
        compression=compression,
        output_types=structure.get_flat_tensor_types(dataset.element_spec),
        output_shapes=structure.get_flat_tensor_shapes(dataset.element_spec),
        )
    super(_SaveDataset, self).__init__(dataset, variant_tensor)

  @property
  def _functions(self):
    return [self._shard_func]


def _set_save_dataset_attributes(dataset, shard_func, path):
  """Sets parameters for SaveDatasetOp and SaveDatasetV2Op."""
  if shard_func is None:
    use_shard_func = False
    shard_func = lambda *x: None  # a dummy function that will not be used
  else:
    use_shard_func = True

  wrapped_func = structured_function.StructuredFunctionWrapper(
      shard_func,
      "save()",
      input_structure=dataset.element_spec,
      add_to_graph=False)

  coder = nested_structure_coder.StructureCoder()
  encoded = coder.encode_structure(dataset.element_spec)
  gfile.MakeDirs(path)
  with gfile.GFile(os.path.join(path, DATASET_SPEC_FILENAME), "wb") as f:
    f.write(encoded.SerializeToString())

  path = ops.convert_to_tensor(path, dtype=dtypes.string, name="path")
  shard_func = wrapped_func.function
  shard_func.add_to_graph(ops.get_default_graph())

  # pylint: disable=protected-access
  dataset._apply_debug_options()
  return dataset, shard_func, use_shard_func, path,


class _LoadDataset(dataset_ops.DatasetSource):
  """A dataset that loads previously saved dataset."""

  def __init__(self, path, element_spec=None, compression=None,
               reader_func=None):

    if reader_func is None:
      reader_func = lambda datasets: datasets.interleave(  # pylint:disable=g-long-lambda
          lambda x: x,
          cycle_length=multiprocessing.cpu_count(),
          num_parallel_calls=dataset_ops.AUTOTUNE)

    self._path = path
    if element_spec is None:
      if not context.executing_eagerly():
        raise ValueError(
            "In graph mode the `element_spec` argument must be provided.")
      with gfile.GFile(os.path.join(path, DATASET_SPEC_FILENAME), "rb") as f:
        encoded_spec = f.read()
      struct_pb = nested_structure_coder.struct_pb2.StructuredValue()
      struct_pb.ParseFromString(encoded_spec)
      coder = nested_structure_coder.StructureCoder()
      spec = coder.decode_proto(struct_pb)
      self._element_spec = spec
    else:
      self._element_spec = element_spec
    self._compression = compression
    self._reader_func = structured_function.StructuredFunctionWrapper(
        reader_func,
        "load()",
        # Dataset of datasets of input elements
        input_structure=dataset_ops.DatasetSpec(
            dataset_ops.DatasetSpec(self._element_spec)))

    variant_tensor = gen_experimental_dataset_ops.load_dataset(
        path,
        reader_func_other_args=self._reader_func.function.captured_inputs,
        compression=compression,
        reader_func=self._reader_func.function,
        **self._flat_structure)
    super(_LoadDataset, self).__init__(variant_tensor)

  def _functions(self):
    return [self._reader_func]

  @property
  def element_spec(self):
    return self._element_spec


@tf_export("data.experimental.load", v1=[])
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


  Note that to load a previously saved dataset, you need to specify
  `element_spec` -- a type signature of the elements of the saved dataset, which
  can be obtained via `tf.data.Dataset.element_spec`. This requirement exists so
  that shape inference of the loaded dataset does not need to perform I/O.

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
      `tf.TypeSpec` saved with the saved dataset is used. This argument needs to
      be provided if the method is executed in graph mode.
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

  return _LoadDataset(
      path=path,
      element_spec=element_spec,
      compression=compression,
      reader_func=reader_func)
