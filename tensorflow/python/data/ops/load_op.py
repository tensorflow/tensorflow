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
"""Implementation of LoadDataset in Python."""
import multiprocessing
import os

from google.protobuf import message
from google.protobuf import text_format
from tensorflow.core.protobuf import snapshot_pb2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.platform import gfile
# TODO(b/238903802): Use TypeSpec serialization methods directly.
from tensorflow.python.saved_model import nested_structure_coder


def _load(path, element_spec, compression, reader_func):
  """Loads dataset from tf.data snapshot."""

  def _get_distributed_snapshot_metadata():
    """Reads the distributed snapshot metadata.

    Returns:
      DistributedSnapshotMetadata if the snapshot is a distributed snapshot.
      Returns None if it is a non-distributed snapshot.
    """
    try:
      with gfile.GFile(os.path.join(path, "snapshot.metadata"), "r") as f:
        return text_format.ParseLines(
            f, snapshot_pb2.DistributedSnapshotMetadata())
    except (text_format.ParseError, message.DecodeError, UnicodeDecodeError):
      return None

  if reader_func is None:
    reader_func = lambda datasets: datasets.interleave(  # pylint:disable=g-long-lambda
        lambda x: x,
        cycle_length=multiprocessing.cpu_count(),
        num_parallel_calls=dataset_ops.AUTOTUNE)

  if element_spec is None:
    with gfile.GFile(
        os.path.join(path, dataset_ops.DATASET_SPEC_FILENAME), "rb") as f:
      encoded_spec = f.read()
    struct_pb = nested_structure_coder.struct_pb2.StructuredValue()
    struct_pb.ParseFromString(encoded_spec)
    spec = nested_structure_coder.decode_proto(struct_pb)
    element_spec = spec

  distributed_snapshot_metadata = _get_distributed_snapshot_metadata()
  if distributed_snapshot_metadata:
    _validate_snapshot(path)
    if compression and compression != distributed_snapshot_metadata.compression:
      raise ValueError(
          f"Failed to load tf.data snapshot at {path}. User specified "
          f"compression {compression}, but the actual compression is "
          f"{distributed_snapshot_metadata.compression}.")

    chunks_dir = os.path.join(path, "chunks")
    chunk_files = [
        os.path.join(chunks_dir, f) for f in gfile.ListDirectory(chunks_dir)]
    dataset = dataset_ops.Dataset.from_tensor_slices(chunk_files)
    # TODO(b/250921378): Use the element_spec from the metadata.
    dataset = dataset.map(
        lambda chunk_file: _SnapshotChunkDataset(  # pylint:disable=g-long-lambda
            chunk_file,
            element_spec=element_spec,
            compression=distributed_snapshot_metadata.compression))
    return reader_func(dataset)
  return _LoadDataset(path, element_spec, compression, reader_func)


class _LoadDataset(dataset_ops.DatasetSource):
  """A dataset that loads previously saved dataset."""

  def __init__(self, path, element_spec, compression, reader_func):
    self._path = path
    self._element_spec = element_spec
    self._compression = compression
    self._reader_func = structured_function.StructuredFunctionWrapper(
        reader_func,
        "load()",
        # Dataset of datasets of input elements
        input_structure=dataset_ops.DatasetSpec(
            dataset_ops.DatasetSpec(self._element_spec)))

    variant_tensor = ged_ops.load_dataset(
        path,
        reader_func_other_args=self._reader_func.function.captured_inputs,
        compression=compression,
        reader_func=self._reader_func.function,
        **self._flat_structure)
    super().__init__(variant_tensor)

  @property
  def element_spec(self):
    return self._element_spec


class _SnapshotChunkDataset(dataset_ops.DatasetSource):
  """A dataset for one chunk file from a tf.data distributed snapshot."""

  def __init__(self, chunk_file, element_spec, compression):
    self._chunk_file = chunk_file
    self._element_spec = element_spec
    variant_tensor = ged_ops.snapshot_chunk_dataset(
        chunk_file,
        compression=compression,
        **self._flat_structure)
    super().__init__(variant_tensor)

  @property
  def element_spec(self):
    return self._element_spec


def _validate_snapshot(path):
  """Validates a tf.data distributed snapshot.

  Args:
    path: Root path of the distributed snapshot.

  Raises:
    ValueError if the snapshot is invalid.
  """

  if not gfile.Exists(path):
    raise ValueError(
        f"Failed to load tf.data snapshot at {path}: The snapshot directory "
        "does not exist.")

  if gfile.Exists(os.path.join(path, "ERROR")):
    with gfile.GFile(os.path.join(path, "ERROR"), "r") as f:
      raise ValueError(
          f"Failed to load tf.data snapshot at {path}. The save job failed to "
          f"write it. Status: {f.read()}")

  if not gfile.Exists(os.path.join(path, "DONE")):
    raise ValueError(
        f"Failed to load tf.data snapshot at {path}. The save job has not "
        "finished writing the snapshot.")
