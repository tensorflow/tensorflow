# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
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

"""Sequence feature."""

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow.data.experimental.core import utils
from tensorflow.data.experimental.core.features import feature as feature_lib
from tensorflow.data.experimental.core.features import features_dict
from tensorflow.data.experimental.core.features import top_level_feature
from tensorflow.data.experimental.core.utils import type_utils

Json = type_utils.Json


class Sequence(top_level_feature.TopLevelFeature):
  """Composite `FeatureConnector` for a `dict` where each value is a list.

  `Sequence` correspond to sequence of `tfds.features.FeatureConnector`. At
  generation time, a list for each of the sequence element is given. The output
  of `tf.data.Dataset` will batch all the elements of the sequence together.

  If the length of the sequence is static and known in advance, it should be
  specified in the constructor using the `length` param.

  Note that `Sequence` does not support features which are of type
  `tf.io.FixedLenSequenceFeature`.

  Example:
  At construction time:

  ```
  tfds.features.Sequence(tfds.features.Image(), length=NB_FRAME)
  ```

  or:

  ```
  tfds.features.Sequence({
      'frame': tfds.features.Image(shape=(64, 64, 3))
      'action': tfds.features.ClassLabel(['up', 'down', 'left', 'right'])
  }, length=NB_FRAME)
  ```

  During data generation:

  ```
  yield {
      'frame': np.ones(shape=(NB_FRAME, 64, 64, 3)),
      'action': ['left', 'left', 'up', ...],
  }
  ```

  Tensor returned by `.as_dataset()`:

  ```
  {
      'frame': tf.Tensor(shape=(NB_FRAME, 64, 64, 3), dtype=tf.uint8),
      'action': tf.Tensor(shape=(NB_FRAME,), dtype=tf.int64),
  }
  ```

  At generation time, you can specify a list of features dict, a dict of list
  values or a stacked numpy array. The lists will automatically be distributed
  into their corresponding `FeatureConnector`.

  """

  def __init__(self, feature, length=None, **kwargs):
    """Construct a sequence dict.

    Args:
      feature: `dict`, the features to wrap
      length: `int`, length of the sequence if static and known in advance
      **kwargs: `dict`, constructor kwargs of `tfds.features.FeaturesDict`
    """
    # Convert {} => FeaturesDict, tf.int32 => Tensor(shape=(), dtype=tf.int32)
    self._feature = features_dict.to_feature(feature)
    self._length = length
    assert not kwargs, 'Json export/import should be updated'
    super(Sequence, self).__init__(**kwargs)

  @property
  def feature(self):
    """The inner feature."""
    return self._feature

  def _add_length_dim(self, tensor_info):
    """Add the length dimension to the given tensor_info."""
    tensor_info = feature_lib.TensorInfo.copy_from(tensor_info)
    tensor_info.shape = (self._length,) + tensor_info.shape
    tensor_info.sequence_rank += 1
    return tensor_info

  def get_tensor_info(self):
    """See base class for details."""
    # Add the additional length dimension to every shape
    tensor_info = self._feature.get_tensor_info()
    return tf.nest.map_structure(self._add_length_dim, tensor_info)

  def get_serialized_info(self):
    """See base class for details."""
    # Add the additional length dimension to every serialized features
    tensor_info = self._feature.get_serialized_info()
    return tf.nest.map_structure(self._add_length_dim, tensor_info)

  def encode_example(self, example_dict):
    # Convert nested dict[list] into list[nested dict]
    sequence_elements = _transpose_dict_list(example_dict)

    # If length is static, ensure that the given length match
    if self._length is not None and len(sequence_elements) != self._length:
      raise ValueError(
          'Input sequence length do not match the defined one. Got {} != '
          '{}'.format(len(sequence_elements), self._length)
      )

    # Empty sequences return empty arrays
    if not sequence_elements:
      def _build_empty_np(serialized_info):
        return np.empty(
            shape=tuple(s if s else 0 for s in serialized_info.shape),
            dtype=serialized_info.dtype.as_numpy_dtype,
        )

      return tf.nest.map_structure(_build_empty_np, self.get_serialized_info())

    # Encode each individual elements
    sequence_elements = [
        self.feature.encode_example(sequence_elem)
        for sequence_elem in sequence_elements
    ]

    # Then convert back list[nested dict] => nested dict[list]
    def _stack_nested(sequence_elements):
      """Recursivelly stack the tensors from the same dict field."""
      if isinstance(sequence_elements[0], dict):
        return {
            # Stack along the first dimension
            k: _stack_nested(sub_sequence)
            for k, sub_sequence in utils.zip_dict(*sequence_elements)
        }
      # Note: As each field can be a nested ragged list, we don't check here
      # that all elements from the list have matching dtype/shape.
      # Checking is done in `example_serializer` when elements
      # are converted to numpy array and stacked togethers.
      return list(sequence_elements)

    return _stack_nested(sequence_elements)

  def _flatten(self, x):
    """See base class for details."""
    if isinstance(x, Sequence):
      return self.feature._flatten(x.feature)  # pylint: disable=protected-access
    return self.feature._flatten(x)  # pylint: disable=protected-access

  def _nest(self, list_x):
    """See base class for details."""
    return self.feature._nest(list_x)  # pylint: disable=protected-access

  def save_metadata(self, *args, **kwargs):
    """See base class for details."""
    self._feature.save_metadata(*args, **kwargs)

  def load_metadata(self, *args, **kwargs):
    """See base class for details."""
    self._feature.load_metadata(*args, **kwargs)

  def __getitem__(self, key):
    """Convenience method to access the underlying features."""
    return self._feature[key]

  def __getattr__(self, key):
    """Allow to access the underlying attributes directly."""
    return getattr(self._feature, key)

  # The __getattr__ method triggers an infinite recursion loop when loading a
  # pickled instance. So we override that name in the instance dict, and remove
  # it when unplickling.
  def __getstate__(self):
    state = self.__dict__.copy()
    state['__getattr__'] = 0
    return state

  def __setstate__(self, state):
    del state['__getattr__']
    self.__dict__.update(state)

  def __repr__(self):
    """Display the feature."""
    inner_feature_repr = feature_lib.get_inner_feature_repr(self._feature)
    if inner_feature_repr.startswith('FeaturesDict('):
      # Minor formatting cleaning: 'Sequence(FeaturesDict({' => 'Sequence({'
      inner_feature_repr = inner_feature_repr[len('FeaturesDict('):-len(')')]
    return '{}({})'.format(type(self).__name__, inner_feature_repr)

  @classmethod
  def from_json_content(cls, value: Json) -> 'Sequence':
    return cls(
        feature=feature_lib.FeatureConnector.from_json(value['feature']),
        length=value['length']
    )

  def to_json_content(self) -> Json:
    return {
        'feature': self.feature.to_json(),
        'length': self._length,
    }


def _np_to_list(elem):
  """Returns list from list, tuple or ndarray."""
  if isinstance(elem, list):
    return elem
  elif isinstance(elem, tuple):
    return list(elem)
  elif isinstance(elem, np.ndarray):
    return list(elem)
  else:
    raise ValueError(
        'Input elements of a sequence should be either a numpy array, a '
        'python list or tuple. Got {}'.format(type(elem)))


def _transpose_dict_list(dict_list):
  """Transpose a nested dict[list] into a list[nested dict]."""
  # 1. Unstack numpy arrays into list
  dict_list = utils.map_nested(_np_to_list, dict_list, dict_only=True)

  # 2. Extract the sequence length (and ensure the length is constant for all
  # elements)
  length = {'value': None}  # dict because `nonlocal` is Python3 only

  def update_length(elem):
    if length['value'] is None:
      length['value'] = len(elem)
    elif length['value'] != len(elem):
      raise ValueError(
          'The length of all elements of one sequence should be the same. '
          'Got {} != {}'.format(length['value'], len(elem)))
    return elem
  utils.map_nested(update_length, dict_list, dict_only=True)

  # 3. Extract each individual elements
  return [
      utils.map_nested(
          lambda elem: elem[i], dict_list, dict_only=True)   # pylint: disable=cell-var-from-loop
      for i in range(length['value'])  # pytype: disable=wrong-arg-types
  ]
