# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Input parsing code for LabeledTensors."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six import string_types

from tensorflow.contrib.labeled_tensor.python.ops import _typecheck as tc
from tensorflow.contrib.labeled_tensor.python.ops import core
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import parsing_ops


class FixedLenFeature(object):
  """Configuration for parsing a fixed-length input feature.

  Fields:
    axes: A list of Axis objects or tuples (axis_name, axis_value),
      where `axis_name` is a string and `axis_value` is None (unknown size), an
      integer or a list of tick labels.
    dtype: Data type of input.
    default_value: Value to be used if an example is missing this feature. It
        must be compatible with `dtype`.
  """

  def __init__(self, axes, dtype, default_value=None):
    self._axes = [core.as_axis(a) for a in axes]
    self._dtype = dtype
    self._default_value = default_value

  @property
  def axes(self):
    return self._axes

  @property
  def dtype(self):
    return self._dtype

  @property
  def default_value(self):
    return self._default_value


@tc.returns(tc.Dict(string_types, parsing_ops.FixedLenFeature))
@tc.accepts(tc.Mapping(string_types, FixedLenFeature))
def _labeled_to_unlabeled_features(features):
  """Convert a dict of lt.FixedLenFeature into a dict of tf.FixedLenFeature."""
  unlabeled_features = {}
  for name, labeled_feature in features.items():
    shape = [ax.size for ax in labeled_feature.axes]
    if any(size is None for size in shape):
      # This should be caught on the TensorFlow side, but it isn't yet:
      # https://github.com/tensorflow/tensorflow/issues/2874
      raise ValueError('axes with unknown size are not supported')
    dtype = labeled_feature.dtype
    default_value = labeled_feature.default_value
    unlabeled_features[name] = parsing_ops.FixedLenFeature(
        shape, dtype, default_value)
  return unlabeled_features


@tc.returns(tc.Dict(string_types, core.LabeledTensor))
@tc.accepts(core.LabeledTensorLike, tc.Mapping(string_types, FixedLenFeature),
            tc.Optional(string_types), object)
def parse_example(serialized, features, name=None, example_names=None):
  """Parse `Example` protos into a `dict` of labeled tensors.

  See tf.parse_example.

  Args:
    serialized: A 1-D LabeledTensor of strings, a batch of binary serialized
      `Example` protos.
    features: A `dict` mapping feature keys to `labeled_tensor.FixedLenFeature`
      values.
    name: A name for this operation (optional).
    example_names: A vector (1-D Tensor) of strings (optional), the names of
      the serialized protos in the batch.

  Returns:
    A `dict` mapping feature keys to `LabeledTensor` values. The single axis
    from `serialized` will be prepended to the axes provided by each feature.

  Raises:
    ValueError: if any feature is invalid.
  """
  serialized = core.convert_to_labeled_tensor(serialized)
  unlabeled_features = _labeled_to_unlabeled_features(features)

  unlabeled_parsed = parsing_ops.parse_example(
      serialized.tensor, unlabeled_features, name, example_names)

  parsed = {}
  for name, parsed_feature in unlabeled_parsed.items():
    axes = list(serialized.axes.values()) + features[name].axes
    parsed[name] = core.LabeledTensor(parsed_feature, axes)

  return parsed


@tc.returns(tc.Dict(string_types, core.LabeledTensor))
@tc.accepts(core.LabeledTensorLike, tc.Mapping(string_types, FixedLenFeature),
            tc.Optional(string_types), object)
def parse_single_example(serialized, features, name=None, example_names=None):
  """Parses a single `Example` proto.

  See tf.parse_single_example.

  Args:
    serialized: A scalar string Tensor or LabeledTensor, a single serialized
      Example.
    features: A `dict` mapping feature keys to `labeled_tensor.FixedLenFeature`
      values.
    name: A name for this operation (optional).
    example_names: (Optional) A scalar string Tensor, the associated name.

  Returns:
    A `dict` mapping feature keys to `LabeledTensor` values.

  Raises:
    ValueError: if any feature is invalid.
  """
  serialized = core.convert_to_labeled_tensor(serialized)
  unlabeled_features = _labeled_to_unlabeled_features(features)

  unlabeled_parsed = parsing_ops.parse_single_example(
      serialized.tensor, unlabeled_features, name, example_names)

  parsed = {}
  for name, parsed_feature in unlabeled_parsed.items():
    parsed[name] = core.LabeledTensor(parsed_feature, features[name].axes)

  return parsed


@tc.returns(core.LabeledTensor)
@tc.accepts(dtypes.DType, tc.Collection(tc.Union(string_types, core.AxisLike)),
            tc.Optional(string_types))
def placeholder(dtype, axes, name=None):
  """Create a placeholder for a labeled tensor.

  For example:

    lt.placeholder(tf.float32, ['batch', ('channel', ['r', 'g', 'b'])])

  See tf.compat.v1.placeholder for more details.

  Args:
    dtype: The type of elements in the tensor to be fed.
    axes: sequence of strings (denoting axes of unknown size) and/or objects
      convertable to lt.Axis to label the result.
    name: Optional op name.

  Returns:
    Placeholder labeled tensor.
  """
  with ops.name_scope(name, 'lt_placeholder', []) as scope:
    axes = core.Axes([(axis, None) if isinstance(axis, string_types) else axis
                      for axis in axes])
    shape = [axis.size for axis in axes.values()]
    tensor = array_ops.placeholder(dtype, shape, name=scope)
    return core.LabeledTensor(tensor, axes)
