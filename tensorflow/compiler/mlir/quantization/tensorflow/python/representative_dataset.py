# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Defines types required for representative datasets for quantization."""

import collections.abc
import os
from typing import Iterable, Mapping, Optional, Union

from tensorflow.compiler.mlir.quantization.tensorflow import quantization_options_pb2
from tensorflow.python.client import session
from tensorflow.python.data.ops import readers
from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import python_io
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import core
from tensorflow.python.util import tf_export

# A representative sample is a map of: input_key -> input_value.
# Ex.: {'dense_input': tf.constant([1, 2, 3])}
# Ex.: {'x1': np.ndarray([4, 5, 6]}
RepresentativeSample = Mapping[str, core.TensorLike]

# A representative dataset is an iterable of representative samples.
RepresentativeDataset = Iterable[RepresentativeSample]

# A type representing a map from: signature key -> representative dataset.
# Ex.: {'serving_default': [tf.constant([1, 2, 3]), tf.constant([4, 5, 6])],
#       'other_signature_key': [tf.constant([[2, 2], [9, 9]])]}
RepresentativeDatasetMapping = Mapping[str, RepresentativeDataset]

# A type alias expressing that it can be either a RepresentativeDataset or
# a mapping of signature key to RepresentativeDataset.
RepresentativeDatasetOrMapping = Union[
    RepresentativeDataset, RepresentativeDatasetMapping
]

# Type aliases for quantization_options_pb2 messages.
_RepresentativeDataSample = quantization_options_pb2.RepresentativeDataSample
_RepresentativeDatasetFile = quantization_options_pb2.RepresentativeDatasetFile


class RepresentativeDatasetSaver:
  """Representative dataset saver.

  Exposes a single method `save` that saves the provided representative dataset
  into files.

  This is useful when you would like to keep a snapshot of your representative
  dataset at a file system or when you need to pass the representative dataset
  as files.
  """

  def save(
      self, representative_dataset: RepresentativeDatasetMapping
  ) -> Mapping[str, _RepresentativeDatasetFile]:
    """Saves the representative dataset.

    Args:
      representative_dataset: RepresentativeDatasetMapping which is a
        signature_def_key -> representative dataset mapping.
    """
    raise NotImplementedError('Method "save" is not implemented.')


@tf_export.tf_export(
    'quantization.experimental.TfRecordRepresentativeDatasetSaver'
)
class TfRecordRepresentativeDatasetSaver(RepresentativeDatasetSaver):
  """Representative dataset saver in TFRecord format.

  Saves representative datasets for quantization calibration in TFRecord format.
  The samples are serialized as `RepresentativeDataSample`.

  The `save` method return a signature key to `RepresentativeDatasetFile` map,
  which can be used for QuantizationOptions.

  Example usage:

  ```python
  # Creating the representative dataset.
  representative_dataset = [{"input": tf.random.uniform(shape=(3, 3))}
                        for _ in range(256)]

  # Saving to a TFRecord file.
  dataset_file_map = (
    tf.quantization.experimental.TfRecordRepresentativeDatasetSaver(
          path_map={'serving_default': '/tmp/representative_dataset_path'}
      ).save({'serving_default': representative_dataset})
  )

  # Using in QuantizationOptions.
  quantization_options = tf.quantization.experimental.QuantizationOptions(
      signature_keys=['serving_default'],
      representative_datasets=dataset_file_map,
  )
  tf.quantization.experimental.quantize_saved_model(
      '/tmp/input_model',
      '/tmp/output_model',
      quantization_options=quantization_options,
  )
  ```
  """

  def __init__(self, path_map: Mapping[str, os.PathLike[str]]):
    """Initializes TFRecord represenatative dataset saver.

    Args:
      path_map: Signature def key -> path mapping. Each path is a TFRecord file
        to which a `RepresentativeDataset` is saved. The signature def keys
        should be a subset of the `SignatureDef` keys of the
        `representative_dataset` argument of the `save()` call.
    """
    self.path_map: Mapping[str, os.PathLike[str]] = path_map

  def _save_tf_record_dataset(
      self,
      repr_ds: RepresentativeDataset,
      signature_def_key: str,
  ) -> _RepresentativeDatasetFile:
    """Saves `repr_ds` to a TFRecord file.

    Each sample in `repr_ds` is serialized as `RepresentativeDataSample`.

    Args:
      repr_ds: `RepresentativeDataset` to save.
      signature_def_key: The signature def key associated with `repr_ds`.

    Returns:
      a RepresentativeDatasetFile instance contains the path to the saved file.
    """
    tfrecord_file_path = self.path_map[signature_def_key]
    with python_io.TFRecordWriter(tfrecord_file_path) as writer:
      for repr_sample in repr_ds:
        sample = _RepresentativeDataSample()
        for input_name, input_value in repr_sample.items():
          sample.tensor_proto_inputs[input_name].CopyFrom(
              tensor_util.make_tensor_proto(input_value)
          )

        writer.write(sample.SerializeToString())

    logging.info(
        'Saved representative dataset for signature def: %s to: %s',
        signature_def_key,
        tfrecord_file_path,
    )
    return _RepresentativeDatasetFile(
        tfrecord_file_path=str(tfrecord_file_path)
    )

  def save(
      self, representative_dataset: RepresentativeDatasetMapping
  ) -> Mapping[str, _RepresentativeDatasetFile]:
    """Saves the representative dataset.

    Args:
      representative_dataset: Signature def key -> representative dataset
        mapping. Each dataset is saved in a separate TFRecord file whose path
        matches the signature def key of `path_map`.

    Raises:
      ValueError: When the signature def key in `representative_dataset` is not
      present in the `path_map`.

    Returns:
      A map from signature key to the RepresentativeDatasetFile instance
      contains the path to the saved file.
    """
    dataset_file_map = {}
    for signature_def_key, repr_ds in representative_dataset.items():
      if signature_def_key not in self.path_map:
        raise ValueError(
            'SignatureDef key does not exist in the provided path_map:'
            f' {signature_def_key}'
        )

      dataset_file_map[signature_def_key] = self._save_tf_record_dataset(
          repr_ds, signature_def_key
      )
    return dataset_file_map


class RepresentativeDatasetLoader:
  """Representative dataset loader.

  Exposes the `load` method that loads the representative dataset from files.
  """

  def __init__(
      self,
      dataset_file_map: Mapping[str, _RepresentativeDatasetFile],
  ) -> None:
    """Initializes TFRecord represenatative dataset loader.

    Args:
      dataset_file_map: Signature key -> `RepresentativeDatasetFile` mapping.

    Raises:
      DecodeError: If the sample is not RepresentativeDataSample.
    """
    self.dataset_file_map = dataset_file_map

  def _load_tf_record(self, tf_record_path: str) -> RepresentativeDataset:
    """Loads TFRecord containing samples of type`RepresentativeDataSample`."""
    samples = []
    with context.eager_mode():
      for sample_bytes in readers.TFRecordDatasetV2(filenames=[tf_record_path]):
        sample_proto = _RepresentativeDataSample.FromString(
            sample_bytes.numpy()
        )
        sample = {}
        for input_key, tensor_proto in sample_proto.tensor_proto_inputs.items():
          sample[input_key] = tensor_util.MakeNdarray(tensor_proto)
        samples.append(sample)
    return samples

  def load(self) -> RepresentativeDatasetMapping:
    """Loads the representative datasets.

    Returns:
      representative dataset mapping: A signature def key -> representative
      mapping. The loader loads `RepresentativeDataset` for each path in
      `self.dataset_file_map` and associates the loaded dataset to the
      corresponding signature def key.
    """
    repr_dataset_map = {}
    for signature_def_key, dataset_file in self.dataset_file_map.items():
      if dataset_file.HasField('tfrecord_file_path'):
        repr_dataset_map[signature_def_key] = self._load_tf_record(
            dataset_file.tfrecord_file_path
        )
      else:
        raise ValueError('Unsupported Representative Dataset filetype')

    return repr_dataset_map


def replace_tensors_by_numpy_ndarrays(
    repr_ds: RepresentativeDataset, sess: session.Session
) -> RepresentativeDataset:
  """Replaces tf.Tensors in samples by their evaluated numpy arrays.

  Note: This should be run in graph mode (default in TF1) only.

  Args:
    repr_ds: Representative dataset to replace the tf.Tensors with their
      evaluated values. `repr_ds` is iterated through, so it may not be reusable
      (e.g. if it is a generator object).
    sess: Session instance used to evaluate tf.Tensors.

  Returns:
    The new representative dataset where each tf.Tensor is replaced by its
    evaluated numpy ndarrays.
  """
  new_repr_ds = []
  for sample in repr_ds:
    new_sample = {}
    for input_key, input_data in sample.items():
      # Evaluate the Tensor to get the actual value.
      if isinstance(input_data, core.Tensor):
        input_data = input_data.eval(session=sess)

      new_sample[input_key] = input_data

    new_repr_ds.append(new_sample)
  return new_repr_ds


def get_num_samples(repr_ds: RepresentativeDataset) -> Optional[int]:
  """Returns the number of samples if known.

  Args:
    repr_ds: Representative dataset.

  Returns:
    Returns the total number of samples in `repr_ds` if it can be determined
    without iterating the entier dataset. Returns None iff otherwise. When it
    returns None it does not mean the representative dataset is infinite or it
    is malformed; it simply means the size cannot be determined without
    iterating the whole dataset.
  """
  if isinstance(repr_ds, collections.abc.Sized):
    try:
      return len(repr_ds)
    except Exception as ex:  # pylint: disable=broad-except
      # There are some cases where calling __len__() raises an exception.
      # Handle this as if the size is unknown.
      logging.info('Cannot determine the size of the dataset (%s).', ex)
      return None
  else:
    return None
