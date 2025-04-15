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

from collections.abc import Collection, Sized
import os
from typing import Iterable, Mapping, Optional, Union

import numpy as np

from tensorflow.compiler.mlir.quantization.tensorflow import quantization_options_pb2
from tensorflow.core.protobuf import meta_graph_pb2
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

  def __init__(
      self,
      path_map: Mapping[str, os.PathLike[str]],
      expected_input_key_map: Optional[Mapping[str, Collection[str]]] = None,
  ):
    """Initializes TFRecord represenatative dataset saver.

    Args:
      path_map: Signature def key -> path mapping. Each path is a TFRecord file
        to which a `RepresentativeDataset` is saved. The signature def keys
        should be a subset of the `SignatureDef` keys of the
        `representative_dataset` argument of the `save()` call.
      expected_input_key_map: Signature def key -> expected input keys. If set,
        validate that the sample has same set of input keys before saving.

    Raises:
      KeyError: If path_map and expected_input_key_map have different keys.
    """
    self.path_map: Mapping[str, os.PathLike[str]] = path_map
    self.expected_input_key_map: Mapping[str, Collection[str]] = {}
    if expected_input_key_map is not None:
      if set(path_map.keys()) != set(expected_input_key_map.keys()):
        raise KeyError(
            'The `path_map` and `expected_input_key_map` should have the same'
            ' set of keys.'
        )

      self.expected_input_key_map = expected_input_key_map

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

    Raises:
      KeyError: If the set of input keys in the dataset samples doesn't match
      the set of expected input keys.
    """
    # When running in graph mode (TF1), tf.Tensor types should be converted to
    # numpy ndarray types to be compatible with `make_tensor_proto`.
    if not context.executing_eagerly():
      with session.Session() as sess:
        repr_ds = replace_tensors_by_numpy_ndarrays(repr_ds, sess)

    expected_input_keys = self.expected_input_key_map.get(
        signature_def_key, None
    )
    tfrecord_file_path = self.path_map[signature_def_key]
    with python_io.TFRecordWriter(tfrecord_file_path) as writer:
      for repr_sample in repr_ds:
        if (
            expected_input_keys is not None
            and set(repr_sample.keys()) != expected_input_keys
        ):
          raise KeyError(
              'Invalid input keys for representative sample. The function'
              f' expects input keys of: {set(expected_input_keys)}. Got:'
              f' {set(repr_sample.keys())}. Please provide correct input keys'
              ' for representative samples.'
          )

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

  def load(self) -> RepresentativeDatasetMapping:
    """Loads the representative datasets.

    Returns:
      representative dataset mapping: A loaded signature def key ->
      representative mapping.
    """
    raise NotImplementedError('Method "load" is not implemented.')


class TfRecordRepresentativeDatasetLoader(RepresentativeDatasetLoader):
  """TFRecord representative dataset loader.

  Loads representative dataset stored in TFRecord files.
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
  if isinstance(repr_ds, Sized):
    try:
      return len(repr_ds)
    except Exception as ex:  # pylint: disable=broad-except
      # There are some cases where calling __len__() raises an exception.
      # Handle this as if the size is unknown.
      logging.info('Cannot determine the size of the dataset (%s).', ex)
      return None
  else:
    return None


def create_feed_dict_from_input_data(
    input_data: RepresentativeSample,
    signature_def: meta_graph_pb2.SignatureDef,
) -> Mapping[str, np.ndarray]:
  """Constructs a feed_dict from input data.

  Note: This function should only be used in graph mode.

  This is a helper function that converts an 'input key -> input value' mapping
  to a feed dict. A feed dict is an 'input tensor name -> input value' mapping
  and can be directly passed to the `feed_dict` argument of `sess.run()`.

  Args:
    input_data: Input key -> input value mapping. The input keys should match
      the input keys of `signature_def`.
    signature_def: A SignatureDef representing the function that `input_data` is
      an input to.

  Returns:
    Feed dict, which is intended to be used as input for `sess.run`. It is
    essentially a mapping: input tensor name -> input value. Note that the input
    value in the feed dict is not a `Tensor`.
  """
  feed_dict = {}
  for input_key, input_value in input_data.items():
    input_tensor_name = signature_def.inputs[input_key].name

    value = input_value
    if isinstance(input_value, core.Tensor):
      # Take the data out of the tensor.
      value = input_value.eval()

    feed_dict[input_tensor_name] = value

  return feed_dict
