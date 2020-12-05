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

"""Feature connector."""

import abc
import collections
import json
import os
from typing import Dict, Type, TypeVar

import numpy as np
import six
import tensorflow.compat.v2 as tf

from tensorflow.data.experimental.core import utils
from tensorflow.data.experimental.core.utils import type_utils

Json = type_utils.Json

T = TypeVar('T', bound='FeatureConnector')


class TensorInfo(object):
  """Structure containing info on the `tf.Tensor` shape/dtype."""

  __slots__ = ['shape', 'dtype', 'default_value', 'sequence_rank']

  def __init__(self, shape, dtype, default_value=None, sequence_rank=None):
    """Constructor.

    Args:
      shape: `tuple[int]`, shape of the tensor
      dtype: Tensor dtype
      default_value: Used for retrocompatibility with previous files if a new
        field is added to provide a default value when reading the file.
      sequence_rank: `int`, Number of `tfds.features.Sequence` dimension.
    """
    self.shape = shape
    self.dtype = dtype
    self.default_value = default_value
    self.sequence_rank = sequence_rank or 0

  @classmethod
  def copy_from(cls, tensor_info):
    """Copy constructor."""
    return cls(
        shape=tensor_info.shape,
        dtype=tensor_info.dtype,
        default_value=tensor_info.default_value,
        sequence_rank=tensor_info.sequence_rank,
    )

  def __eq__(self, other):
    """Equality."""
    return (
        self.shape == other.shape and
        self.dtype == other.dtype and
        self.default_value == other.default_value
    )

  def __repr__(self):
    return '{}(shape={}, dtype={})'.format(
        type(self).__name__,
        self.shape,
        repr(self.dtype),
    )


@six.add_metaclass(abc.ABCMeta)
class FeatureConnector(object):
  """Abstract base class for feature types.

  This class provides an interface between the way the information is stored
  on disk, and the way it is presented to the user.

  Here is a diagram on how FeatureConnector methods fit into the data
  generation/reading:

  ```
  generator => encode_example() => tf_example => decode_example() => data dict
  ```

  The connector can either get raw or dictionary values as input, depending on
  the connector type.

  """

  # Keep track of all sub-classes.
  _registered_features: Dict[str, 'FeatureConnector'] = {}

  def __init_subclass__(cls):
    """Registers subclasses features."""
    cls._registered_features[f'{cls.__module__}.{cls.__name__}'] = cls

  @abc.abstractmethod
  def get_tensor_info(self):
    """Return the tf.Tensor dtype/shape of the feature.

    This returns the tensor dtype/shape, as returned by .as_dataset by the
    `tf.data.Dataset` object.

    Ex:

    ```
    return {
        'image': tfds.features.TensorInfo(shape=(None,), dtype=tf.uint8),
        'height': tfds.features.TensorInfo(shape=(), dtype=tf.int32),
        'width': tfds.features.TensorInfo(shape=(), dtype=tf.int32),
    }
    ```

    FeatureConnector which are not containers should return the feature proto
    directly:

    ```
    return tfds.features.TensorInfo(shape=(256, 256), dtype=tf.uint8)
    ```

    Returns:
      tensor_info: Either a dict of `tfds.features.TensorInfo` object, or a
        `tfds.features.TensorInfo`

    """
    raise NotImplementedError

  @property
  def shape(self):
    """Return the shape (or dict of shape) of this FeatureConnector."""
    return tf.nest.map_structure(lambda t: t.shape, self.get_tensor_info())

  @property
  def dtype(self):
    """Return the dtype (or dict of dtype) of this FeatureConnector."""
    return tf.nest.map_structure(lambda t: t.dtype, self.get_tensor_info())

  @classmethod
  def from_json(cls, value: Json) -> 'FeatureConnector':
    """FeatureConnector factory.

    This function should be called from the `tfds.features.FeatureConnector`
    base class. Subclass should implement the `from_json_content`.

    Example:

    ```py
    feature = tfds.features.FeatureConnector.from_json(
        {'type': 'Image', 'content': {'shape': [32, 32, 3], 'dtype': 'uint8'}}
    )
    assert isinstance(feature, tfds.features.Image)
    ```

    Args:
      value: `dict(type=, content=)` containing the feature to restore.
        Match dict returned by `to_json`.

    Returns:
      The reconstructed FeatureConnector.
    """
    subclass = cls._registered_features.get(value['type'])
    if subclass is None:
      raise ValueError(
          f'Unrecognized FeatureConnector type: {value["type"]}\n'
          f'Supported: {list(cls._registered_features)}'
      )
    return subclass.from_json_content(value['content'])

  def to_json(self) -> Json:
    # pylint: disable=line-too-long
    """Exports the FeatureConnector to Json.

    Each feature is serialized as a `dict(type=..., content=...)`.

    * `type`: The cannonical name of the feature (`module.FeatureName`).
    * `content`: is specific to each feature connector and defined in
      `to_json_content`. Can contain nested sub-features (like for
      `tfds.features.FeaturesDict` and `tfds.features.Sequence`).

    For example:

    ```python
    tfds.features.FeaturesDict({
        'input': tfds.features.Image(),
        'target': tfds.features.ClassLabel(num_classes=10),
    })
    ```

    Is serialized as:

    ```json
    {
        "type": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
        "content": {
            "input": {
                "type": "tensorflow_datasets.core.features.image_feature.Image",
                "content": {
                    "shape": [null, null, 3],
                    "dtype": "uint8",
                    "encoding_format": "png"
                }
            },
            "target": {
                "type": "tensorflow_datasets.core.features.class_label_feature.ClassLabel",
                "num_classes": 10
            }
        }
    }
    ```

    Returns:
      A `dict(type=, content=)`. Will be forwarded to
        `from_json` when reconstructing the feature.
    """
    # pylint: enable=line-too-long
    return {
        'type': f'{type(self).__module__}.{type(self).__name__}',
        'content': self.to_json_content(),
    }

  @classmethod
  def from_json_content(cls: Type[T], value: Json) -> T:
    """FeatureConnector factory (to overwrite).

    Subclasses should overwritte this method. importing
    the feature connector from the config.

    This function should not be called directly. `FeatureConnector.from_json`
    should be called instead.

    This function  See existing FeatureConnector for
    example of implementation.

    Args:
      value: FeatureConnector information. Match the `dict` returned by
        `to_json_content`.

    Returns:
      The reconstructed FeatureConnector.
    """
    # Should this be an abstract method once user features have been updated ?
    return cls(**value)  # pytype: disable=not-instantiable

  def to_json_content(self) -> Json:
    """FeatureConnector factory (to overwrite).

    This function should be overwritten by the subclass to allow re-importing
    the feature connector from the config. See existing FeatureConnector for
    example of implementation.

    Returns:
      Dict containing the FeatureConnector metadata. Will be forwarded to
        `from_json_content` when reconstructing the feature.
    """
    return dict()

  def save_config(self, root_dir: str) -> None:
    """Exports the `FeatureConnector` to a file.

    Args:
      root_dir: `path/to/dir` containing the `features.json`
    """
    with tf.io.gfile.GFile(make_config_path(root_dir), 'w') as f:
      f.write(json.dumps(self.to_json(), indent=4))
    self.save_metadata(root_dir, feature_name=None)

  @classmethod
  def from_config(cls, root_dir: str) -> 'FeatureConnector':
    """Reconstructs the FeatureConnector from the config file.

    Usage:

    ```
    features = FeatureConnector.from_config('path/to/features.json')
    ```

    Args:
      root_dir: Directory containing to the features.json file.

    Returns:
      The reconstructed feature instance.
    """
    with tf.io.gfile.GFile(make_config_path(root_dir)) as f:
      feature = FeatureConnector.from_json(json.loads(f.read()))
    feature.load_metadata(root_dir, feature_name=None)
    return feature

  def get_serialized_info(self):
    """Return the shape/dtype of features after encoding (for the adapter).

    The `FileAdapter` then use those information to write data on disk.

    This function indicates how this feature is encoded on file internally.
    The DatasetBuilder are written on disk as tf.train.Example proto.

    Ex:

    ```
    return {
        'image': tfds.features.TensorInfo(shape=(None,), dtype=tf.uint8),
        'height': tfds.features.TensorInfo(shape=(), dtype=tf.int32),
        'width': tfds.features.TensorInfo(shape=(), dtype=tf.int32),
    }
    ```

    FeatureConnector which are not containers should return the feature proto
    directly:

    ```
    return tfds.features.TensorInfo(shape=(64, 64), tf.uint8)
    ```

    If not defined, the retuned values are automatically deduced from the
    `get_tensor_info` function.

    Returns:
      features: Either a dict of feature proto object, or a feature proto object

    """
    return self.get_tensor_info()

  @abc.abstractmethod
  def encode_example(self, example_data):
    """Encode the feature dict into tf-example compatible input.

    The input example_data can be anything that the user passed at data
    generation. For example:

    For features:

    ```
    features={
        'image': tfds.features.Image(),
        'custom_feature': tfds.features.CustomFeature(),
    }
    ```

    At data generation (in `_generate_examples`), if the user yields:

    ```
    yield {
        'image': 'path/to/img.png',
        'custom_feature': [123, 'str', lambda x: x+1]
    }
    ```

    Then:

     * `tfds.features.Image.encode_example` will get `'path/to/img.png'` as
       input
     * `tfds.features.CustomFeature.encode_example` will get `[123, 'str',
       lambda x: x+1] as input

    Args:
      example_data: Value or dictionary of values to convert into tf-example
        compatible data.

    Returns:
      tfexample_data: Data or dictionary of data to write as tf-example. Data
        can be a list or numpy array.
        Note that numpy arrays are flattened so it's the feature connector
        responsibility to reshape them in `decode_example()`.
        Note that tf.train.Example only supports int64, float32 and string so
        the data returned here should be integer, float or string. User type
        can be restored in `decode_example()`.
    """
    raise NotImplementedError

  def decode_example(self, tfexample_data):
    """Decode the feature dict to TF compatible input.

    Note: If eager is not enabled, this function will be executed as a
    tensorflow graph (in `tf.data.Dataset.map(features.decode_example)`).

    Args:
      tfexample_data: Data or dictionary of data, as read by the tf-example
        reader. It correspond to the `tf.Tensor()` (or dict of `tf.Tensor()`)
        extracted from the `tf.train.Example`, matching the info defined in
        `get_serialized_info()`.

    Returns:
      tensor_data: Tensor or dictionary of tensor, output of the tf.data.Dataset
        object
    """
    return tfexample_data

  def decode_batch_example(self, tfexample_data):
    """Decode multiple features batched in a single tf.Tensor.

    This function is used to decode features wrapped in
    `tfds.features.Sequence()`.
    By default, this function apply `decode_example` on each individual
    elements using `tf.map_fn`. However, for optimization, features can
    overwrite this method to apply a custom batch decoding.

    Args:
      tfexample_data: Same `tf.Tensor` inputs as `decode_example`, but with
        and additional first dimension for the sequence length.

    Returns:
      tensor_data: Tensor or dictionary of tensor, output of the tf.data.Dataset
        object
    """
    # Note: This all works fine in Eager mode (without tf.function) because
    # tf.data pipelines are always executed in Graph mode.

    # Apply the decoding to each of the individual distributed features.
    return tf.map_fn(
        self.decode_example,
        tfexample_data,
        fn_output_signature=self.dtype,
        parallel_iterations=10,
        name='sequence_decode',
    )

  def decode_ragged_example(self, tfexample_data):
    """Decode nested features from a tf.RaggedTensor.

    This function is used to decode features wrapped in nested
    `tfds.features.Sequence()`.
    By default, this function apply `decode_batch_example` on the flat values
    of the ragged tensor. For optimization, features can
    overwrite this method to apply a custom batch decoding.

    Args:
      tfexample_data: `tf.RaggedTensor` inputs containing the nested encoded
        examples.

    Returns:
      tensor_data: The decoded `tf.RaggedTensor` or dictionary of tensor,
        output of the tf.data.Dataset object
    """
    return tf.ragged.map_flat_values(self.decode_batch_example, tfexample_data)

  def repr_html(self, ex: np.ndarray) -> str:
    """Returns the HTML str representation of the object."""
    return _repr_html(ex)

  def repr_html_batch(self, ex: np.ndarray) -> str:
    """Returns the HTML str representation of the object (Sequence)."""
    return _repr_html(ex)

  def repr_html_ragged(self, ex: np.ndarray) -> str:
    """Returns the HTML str representation of the object (Nested sequence)."""
    return _repr_html(ex)

  def _flatten(self, x):
    """Flatten the input dict into a list of values.

    For instance, the following feature:
    ```
    feature = FeatureDict({
        'a': w,
        'b': x,
        'c': {
            'd': y,
            'e': z,
        },
    })
    ```

    Applied to the following `dict`:
    ```
    feature._flatten({
        'b': X,
        'c': {
            'd': Y,
        },
    })
    ```

    Will produce the following flattened output:
    ```
    [
        None,
        X,
        Y,
        None,
    ]
    ```

    Args:
      x: A nested `dict` like structure matching the structure of the
      `FeatureConnector`. Note that some elements may be missing.

    Returns:
      `list`: The flattened list of element of `x`. Order is guaranteed to be
      deterministic. Missing elements will be filled with `None`.
    """
    return [x]

  def _nest(self, list_x):
    """Pack the list into a nested dict.

    This is the reverse function of flatten.

    For instance, the following feature:
    ```
    feature = FeatureDict({
        'a': w,
        'b': x,
        'c': {
            'd': y,
            'e': z,
        },
    })
    ```

    Applied to the following `dict`:
    ```
    feature._nest([
        None,
        X,
        Y,
        None,
    ])
    ```

    Will produce the following flattened output:
    ```
    {
        'a': None,
        'b': X,
        'c': {
            'd': Y,
            'e': None,
        },
    }
    ```

    Args:
      list_x: List of values matching the flattened `FeatureConnector`
        structure. Missing values should be filled with None.

    Returns:
      nested_x: nested `dict` matching the flattened `FeatureConnector`
        structure.
    """
    assert len(list_x) == 1
    return list_x[0]

  def _additional_repr_info(self):
    """Override to return additional info to go into __repr__."""
    return {}

  def __repr__(self):
    """Display the feature dictionary."""
    tensor_info = self.get_tensor_info()
    if not isinstance(tensor_info, TensorInfo):
      return '{}({})'.format(type(self).__name__, tensor_info)

    # Ensure ordering of keys by adding them one-by-one
    repr_info = collections.OrderedDict()
    repr_info['shape'] = tensor_info.shape
    repr_info['dtype'] = repr(tensor_info.dtype)
    additional_info = self._additional_repr_info()
    for k, v in additional_info.items():
      repr_info[k] = v

    info_str = ', '.join(['%s=%s' % (k, v) for k, v in repr_info.items()])
    return '{}({})'.format(
        type(self).__name__,
        info_str,
    )

  def save_metadata(self, data_dir, feature_name):
    """Save the feature metadata on disk.

    This function is called after the data has been generated (by
    `_download_and_prepare`) to save the feature connector info with the
    generated dataset.

    Some dataset/features dynamically compute info during
    `_download_and_prepare`. For instance:

     * Labels are loaded from the downloaded data
     * Vocabulary is created from the downloaded data
     * ImageLabelFolder compute the image dtypes/shape from the manual_dir

    After the info have been added to the feature, this function allow to
    save those additional info to be restored the next time the data is loaded.

    By default, this function do not save anything, but sub-classes can
    overwrite the function.

    Args:
      data_dir: `str`, path to the dataset folder to which save the info (ex:
        `~/datasets/cifar10/1.2.0/`)
      feature_name: `str`, the name of the feature (from the FeaturesDict key)
    """
    pass

  def load_metadata(self, data_dir, feature_name):
    """Restore the feature metadata from disk.

    If a dataset is re-loaded and generated files exists on disk, this function
    will restore the feature metadata from the saved file.

    Args:
      data_dir: `str`, path to the dataset folder to which save the info (ex:
        `~/datasets/cifar10/1.2.0/`)
      feature_name: `str`, the name of the feature (from the FeaturesDict key)
    """
    pass


class Tensor(FeatureConnector):
  """`FeatureConnector` for generic data of arbitrary shape and type."""

  def __init__(self, *, shape, dtype):
    """Construct a Tensor feature."""
    self._shape = tuple(shape)
    self._dtype = dtype

  def get_tensor_info(self):
    """See base class for details."""
    return TensorInfo(shape=self._shape, dtype=self._dtype)

  def decode_batch_example(self, example_data):
    """See base class for details."""
    # Overwrite the `tf.map_fn`, decoding is a no-op
    return self.decode_example(example_data)

  def decode_ragged_example(self, example_data):
    """See base class for details."""
    # Overwrite the `tf.map_fn`, decoding is a no-op
    return self.decode_example(example_data)

  def encode_example(self, example_data):
    """See base class for details."""
    np_dtype = np.dtype(self.dtype.as_numpy_dtype)
    if isinstance(example_data, tf.Tensor):
      raise TypeError(
          f'Error encoding: {example_data!r}. `_generate_examples` should '
          'yield `np.array` compatible values, not `tf.Tensor`'
      )
    if not isinstance(example_data, np.ndarray):
      example_data = np.array(example_data, dtype=np_dtype)
    # Ensure the shape and dtype match
    if example_data.dtype != np_dtype:
      raise ValueError('Dtype {} do not match {}'.format(
          example_data.dtype, np_dtype))
    utils.assert_shape_match(example_data.shape, self._shape)
    return example_data

  @classmethod
  def from_json_content(cls, value: Json) -> 'Tensor':
    shape = tuple(value['shape'])
    dtype = tf.dtypes.as_dtype(value['dtype'])
    return cls(shape=shape, dtype=dtype)

  def to_json_content(self) -> Json:
    return {
        'shape': list(self._shape),
        'dtype': self._dtype.name,
    }


def make_config_path(root_dir: str) -> str:
  """Returns the path to the features config."""
  return os.path.join(root_dir, 'features.json')


def get_inner_feature_repr(feature):
  """Utils which returns the object which should get printed in __repr__.

  This is used in container features (Sequence, FeatureDict) to print scalar
  Tensor in a less verbose way `Sequence(tf.int32)` rather than
  `Sequence(Tensor(shape=(), dtype=tf.in32))`.

  Args:
    feature: The feature to dispaly

  Returns:
    Either the feature or it's inner value.
  """
  # We only print `tf.int32` rather than `Tensor(shape=(), dtype=tf.int32)`
  # * For the base `Tensor` class (and not subclass).
  # * When shape is scalar (explicit check to avoid trigger when `shape=None`).
  if type(feature) == Tensor and feature.shape == ():  # pylint: disable=unidiomatic-typecheck,g-explicit-bool-comparison
    return repr(feature.dtype)
  else:
    return repr(feature)


def _repr_html(ex) -> str:
  """Default HTML repr."""
  if isinstance(ex, np.ndarray) and ex.size > 1:
    # Do not print individual values for array as it is slow
    # TODO(tfds): We could display a snippet, like the first/last tree items
    return f'{type(ex).__qualname__}(shape={ex.shape}, dtype={ex.dtype})'
  return repr(ex)
