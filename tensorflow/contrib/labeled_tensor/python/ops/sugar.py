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

"""Tools to make it a bit easier to use LabeledTensor."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six import string_types

from tensorflow.contrib.labeled_tensor.python.ops import _typecheck as tc
from tensorflow.contrib.labeled_tensor.python.ops import core
from tensorflow.contrib.labeled_tensor.python.ops import ops
from tensorflow.python.framework import ops as tf_ops


class ReshapeCoder(object):
  """Utility class for mapping to and from another shape.

  For example, say you have a function `crop_center` which expects a
  LabeledTensor with axes named ['batch', 'row', 'column', 'depth'], and
  you have a LabeledTensor `masked_image_lt` with axes ['batch', 'row',
  'column', 'channel', 'mask'].

  To call `crop_center` with `masked_image_lt` you'd normally have to write:

  >>> reshape_lt = lt.reshape(masked_image_lt, ['channel', 'mask'], ['depth'])
  >>> crop_lt = crop_center(reshape_lt)
  >>> result_lt = lt.reshape(crop_lt, ['depth'],
  ...   [masked_image_lt.axes['channel'], masked_image_lt.axes['mask']])

  ReshapeCoder takes care of this renaming logic for you, allowing you to
  instead write:

  >>> rc = ReshapeCoder(['channel', 'mask'], ['depth'])
  >>> result_lt = rc.decode(crop_center(rc.encode(masked_image_lt)))

  Here, `decode` restores the original axes 'channel' and 'mask', so
  `crop_center` must not have modified the size of the 'depth' axis.
  """

  @tc.accepts(object, tc.Collection(str),
              tc.Collection(tc.Union(str, core.AxisLike)), tc.Optional(str))
  def __init__(self, existing_axis_names, new_axes, name=None):
    self._name = name
    self._existing_axis_names = existing_axis_names
    self._new_axes = new_axes

    self._existing_axes = None

  @tc.returns(core.LabeledTensor)
  @tc.accepts(object, core.LabeledTensorLike)
  def encode(self, labeled_tensor):
    """Reshape the input to the target shape.

    If called several times, the axes named in existing_axis_names must be
    identical.

    Args:
      labeled_tensor: The input tensor.

    Returns:
      The input reshaped to the target shape.

    Raises:
      ValueError: If the axes in existing_axis_names don't match the axes of
        a tensor in a previous invocation of this method.
    """
    with tf_ops.name_scope(self._name, 'lt_reshape_encode',
                           [labeled_tensor]) as scope:
      labeled_tensor = core.convert_to_labeled_tensor(labeled_tensor)

      reshape_lt = ops.reshape(labeled_tensor,
                               self._existing_axis_names,
                               self._new_axes,
                               name=scope)

      axes = [labeled_tensor.axes[n] for n in self._existing_axis_names]
      if self._existing_axes is not None and self._existing_axes != axes:
        raise ValueError(
            'input axes %r do not match axes from previous method call %r' %
            (axes, self._existing_axes))
      else:
        self._existing_axes = axes

      return reshape_lt

  @tc.returns(core.LabeledTensor)
  @tc.accepts(object, core.LabeledTensorLike)
  def decode(self, labeled_tensor):
    """Reshape the input to the original shape.

    This is the inverse of encode.
    Encode must have been called at least once prior to this method being
    called.

    Args:
      labeled_tensor: The input tensor.

    Returns:
      The input reshaped to the original shape.

    Raises:
      ValueError: If this method was called before encode was called.
    """
    if self._existing_axes is None:
      raise ValueError('decode called before encode')

    with tf_ops.name_scope(self._name, 'lt_reshape_decode',
                           [labeled_tensor]) as scope:
      labeled_tensor = core.convert_to_labeled_tensor(labeled_tensor)

      new_axis_names = [axis if isinstance(axis, string_types) else
                        core.as_axis(axis).name for axis in self._new_axes]

      return ops.reshape(labeled_tensor,
                         new_axis_names,
                         self._existing_axes,
                         name=scope)
