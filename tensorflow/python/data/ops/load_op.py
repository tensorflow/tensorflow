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
import time
from typing import Any, Callable, Optional, Union

from absl import logging

from google.protobuf import message
from google.protobuf import text_format
from tensorflow.core.protobuf import snapshot_pb2
from tensorflow.python.data.experimental.service import _pywrap_snapshot_utils
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.platform import gfile
# TODO(b/238903802): Use TypeSpec serialization methods directly.
from tensorflow.python.saved_model import nested_structure_coder

# For distributed snapshot load V2, if the snapshot does not exist in this time,
# wait and retry. Raises an ValueError on timeout.
_LOAD_TIMEOUT_SECONDS = 1800


def _load(  # pylint: disable=unused-private-name
    path: str,
    element_spec: Any,
    compression: Optional[str],
    reader_func: Optional[Callable[[dataset_ops.Dataset], dataset_ops.Dataset]]
) -> dataset_ops.Dataset:
  """Loads dataset from tf.data snapshot."""

  if reader_func is None:
    reader_func = lambda datasets: datasets.interleave(  # pylint:disable=g-long-lambda
        lambda x: x,
        cycle_length=multiprocessing.cpu_count(),
        num_parallel_calls=dataset_ops.AUTOTUNE)

  distributed_snapshot_metadata = _load_distributed_snapshot_metadata(path)
  if distributed_snapshot_metadata:
    _validate_snapshot(
        path, distributed_snapshot_metadata, element_spec, compression)
    return _load_distributed_snapshot(
        path, distributed_snapshot_metadata, reader_func)

  if element_spec is None:
    element_spec = _load_element_spec(path)
  return _LoadDataset(path, element_spec, compression, reader_func)


def _load_with_retry(  # pylint: disable=unused-private-name
    path: str,
    element_spec: Any = None,
    compression: Optional[str] = None,
    reader_func: Optional[
        Callable[[dataset_ops.Dataset], dataset_ops.Dataset]] = None,
) -> dataset_ops.Dataset:
  """Tries loading the snapshot. Retries if not found with a timeout."""

  deadline = time.time() + _LOAD_TIMEOUT_SECONDS
  error = None
  while time.time() < deadline:
    try:
      dataset = dataset_ops.Dataset.load(
          path, element_spec, compression, reader_func)
      logging.info("Load tf.data snapshot at %s.", path)
      return dataset
    except errors.NotFoundError as e:
      logging.info(
          "Could not find tf.data snapshot at %s. Will wait and retry.", path)
      error = e
      time.sleep(10)
  raise error


def _load_distributed_snapshot_metadata(
    path: str,
) -> Optional[snapshot_pb2.DistributedSnapshotMetadata]:
  """Reads the distributed snapshot metadata.

  Args:
    path: Base path of the snapshot.

  Returns:
    DistributedSnapshotMetadata if the snapshot is a distributed snapshot.
    Returns None if it is a non-distributed snapshot.
  """
  try:
    with gfile.GFile(
        _pywrap_snapshot_utils.TF_DATA_SnapshotMetadataFilePath(path), "r"
    ) as f:
      return text_format.ParseLines(
          f, snapshot_pb2.DistributedSnapshotMetadata())
  except (
      errors.NotFoundError,
      text_format.ParseError,
      message.DecodeError,
      UnicodeDecodeError):
    return None


def _load_distributed_snapshot(
    path: str,
    metadata: snapshot_pb2.DistributedSnapshotMetadata,
    reader_func: Callable[[dataset_ops.Dataset], dataset_ops.Dataset],
) -> dataset_ops.Dataset:
  """Loads a distributed snapshot."""

  dataset = _ListSnapshotChunksDataset(path)
  dataset = dataset.map(
      lambda chunk_file: _SnapshotChunkDataset(  # pylint:disable=g-long-lambda
          chunk_file,
          element_spec=_parse_element_spec(metadata.element_spec),
          compression=metadata.compression))
  return reader_func(dataset)


def _load_element_spec(path: str) -> Any:
  """Loads the dataset element spec.

  Args:
    path: Base path of the snapshot.

  Returns:
    Dataset element_spec.
  """
  with gfile.GFile(
      os.path.join(path, dataset_ops.DATASET_SPEC_FILENAME), "rb") as f:
    encoded_spec = f.read()
  return _parse_element_spec(encoded_spec)


def _parse_element_spec(encoded_element_spec: Union[bytes, str]) -> Any:
  struct_pb = nested_structure_coder.struct_pb2.StructuredValue()
  struct_pb.ParseFromString(encoded_element_spec)
  return nested_structure_coder.decode_proto(struct_pb)


class _LoadDataset(dataset_ops.DatasetSource):
  """A dataset that loads previously saved dataset."""

  def __init__(
      self,
      path: str,
      element_spec: Any,
      compression: str,
      reader_func: Callable[[dataset_ops.Dataset], dataset_ops.Dataset]):
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
  def element_spec(self) -> Any:
    return self._element_spec


class _SnapshotChunkDataset(dataset_ops.DatasetSource):
  """A dataset for one chunk file from a tf.data distributed snapshot."""

  def __init__(self, chunk_file: str, element_spec: Any, compression: str):
    self._chunk_file = chunk_file
    self._element_spec = element_spec
    variant_tensor = ged_ops.snapshot_chunk_dataset(
        chunk_file,
        compression=compression,
        **self._flat_structure)
    super().__init__(variant_tensor)

  @property
  def element_spec(self) -> Any:
    return self._element_spec


class _ListSnapshotChunksDataset(dataset_ops.DatasetSource):
  """A dataset for listing snapshot chunk files.

  It supports listing partially written snapshots. When a snapshot is being
  written, it returns the currently available chunk files.
  """

  def __init__(self, snapshot_path: str):
    self._snapshot_path = snapshot_path
    variant_tensor = ged_ops.list_snapshot_chunks_dataset(
        snapshot_path, **self._flat_structure)
    super().__init__(variant_tensor)

  @property
  def element_spec(self) -> tensor_spec.TensorSpec:
    return tensor_spec.TensorSpec([], dtypes.string)


def _validate_snapshot(
    path: str,
    metadata: snapshot_pb2.DistributedSnapshotMetadata,
    element_spec: Any,
    compression: str) -> None:
  """Validates a tf.data distributed snapshot.

  Args:
    path: Root path of the distributed snapshot.
    metadata: The DistributedSnapshotMetadata of the snapshot.
    element_spec: Dataset element_spec.
    compression: Compression method used for saving.

  Raises:
    ValueError if the snapshot is invalid.
  """

  error_file = _pywrap_snapshot_utils.TF_DATA_SnapshotErrorFilePath(path)
  if gfile.Exists(error_file):
    with gfile.GFile(error_file, "r") as f:
      raise ValueError(
          f"Failed to load tf.data snapshot at {path}. The save job failed to "
          f"write it. Status: {f.read()}")

  snapshot_element_spec = _parse_element_spec(metadata.element_spec)
  if element_spec and element_spec != snapshot_element_spec:
    raise ValueError(
        f"Failed to load tf.data snapshot at {path}. User specified "
        f"element_spec {element_spec}, but the actual element_spec is "
        f"{snapshot_element_spec}.")

  if compression and compression != metadata.compression:
    raise ValueError(
        f"Failed to load tf.data snapshot at {path}. User specified "
        f"compression {compression}, but the actual compression is "
        f"{metadata.compression}.")
