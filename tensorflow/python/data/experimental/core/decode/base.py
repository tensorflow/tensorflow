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

"""Base decoders.
"""

import abc
import functools

import six
import tensorflow.compat.v2 as tf


@six.add_metaclass(abc.ABCMeta)
class Decoder(object):
  """Base decoder object.

  `tfds.decode.Decoder` allows for overriding the default decoding by
  implementing a subclass, or skipping it entirely with
  `tfds.decode.SkipDecoding`.

  Instead of subclassing, you can also create a `Decoder` from a function
  with the `tfds.decode.make_decoder` decorator.

  All decoders must derive from this base class. The implementation can
  access the `self.feature` property which will correspond to the
  `FeatureConnector` to which this decoder is applied.

  To implement a decoder, the main method to override is `decode_example`,
  which takes the serialized feature as input and returns the decoded feature.

  If `decode_example` changes the output dtype, you must also override
  the `dtype` property. This enables compatibility with
  `tfds.features.Sequence`.
  """

  def __init__(self):
    self.feature = None

  def setup(self, *, feature):
    """Transformation contructor.

    The initialization of decode object is deferred because the objects only
    know the builder/features on which it is used after it has been
    constructed, the initialization is done in this function.

    Args:
      feature: `tfds.features.FeatureConnector`, the feature to which is applied
        this transformation.

    """
    self.feature = feature

  @property
  def dtype(self):
    """Returns the `dtype` after decoding."""
    tensor_info = self.feature.get_tensor_info()
    return tf.nest.map_structure(lambda t: t.dtype, tensor_info)

  @abc.abstractmethod
  def decode_example(self, serialized_example):
    """Decode the example feature field (eg: image).

    Args:
      serialized_example: `tf.Tensor` as decoded, the dtype/shape should be
        identical to `feature.get_serialized_info()`

    Returns:
      example: Decoded example.
    """
    raise NotImplementedError('Abstract class')

  def decode_batch_example(self, serialized_example):
    """See `FeatureConnector.decode_batch_example` for details."""
    return tf.map_fn(
        self.decode_example,
        serialized_example,
        dtype=self.dtype,
        parallel_iterations=10,
        name='sequence_decode',
    )


class SkipDecoding(Decoder):
  """Transformation which skip the decoding entirelly.

  Example of usage:

  ```python
  ds = ds.load(
      'imagenet2012',
      split='train',
      decoders={
          'image': tfds.decode.SkipDecoding(),
      }
  )

  for ex in ds.take(1):
    assert ex['image'].dtype == tf.string
  ```
  """

  @property
  def dtype(self):
    tensor_info = self.feature.get_serialized_info()
    return tf.nest.map_structure(lambda t: t.dtype, tensor_info)

  def decode_example(self, serialized_example):
    """Forward the serialized feature field."""
    return serialized_example


class DecoderFn(Decoder):
  """Decoder created by `tfds.decoder.make_decoder` decorator."""

  def __init__(self, fn, output_dtype, *args, **kwargs):
    super(DecoderFn, self).__init__()
    self._fn = fn
    self._output_dtype = output_dtype
    self._args = args
    self._kwargs = kwargs

  @property
  def dtype(self):
    if self._output_dtype is None:
      return super(DecoderFn, self).dtype
    else:
      return self._output_dtype

  def decode_example(self, serialized_example):
    """Decode the example using the function."""
    return self._fn(
        serialized_example, self.feature, *self._args, **self._kwargs)


def make_decoder(output_dtype=None):
  """Decorator to create a decoder.

  The decorated function should have the signature `(example, feature, *args,
  **kwargs) -> decoded_example`.

   * `example`: Serialized example before decoding
   * `feature`: `FeatureConnector` associated with the example
   * `*args, **kwargs`: Optional additional kwargs forwarded to the function

  Example:

  ```
  @tfds.decode.make_decoder(output_dtype=tf.string)
  def no_op_decoder(example, feature):
    \"\"\"Decoder simply decoding feature normally.\"\"\"
    return feature.decode_example(example)

  tfds.load('mnist', split='train', decoders: {
      'image': no_op_decoder(),
  })
  ```

  Args:
    output_dtype: The output dtype after decoding. Required only if the decoded
      example has a different type than the `FeatureConnector.dtype` and is
      used to decode features inside sequences (ex: videos)

  Returns:
    The decoder object
  """  # pylint: disable=g-docstring-has-escape

  def decorator(fn):

    @functools.wraps(fn)
    def decorated(*args, **kwargs):
      return DecoderFn(fn, output_dtype, *args, **kwargs)
    return decorated

  return decorator
