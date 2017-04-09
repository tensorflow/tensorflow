# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================

# pylint: disable=unused-import,g-bad-import-order
"""Contains layer utilies for input validation and format conversion.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np

from tensorflow.python.ops import variables
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util


def convert_data_format(data_format, ndim):
  if data_format == 'channels_last':
    if ndim == 3:
      return 'NWC'
    elif ndim == 4:
      return 'NHWC'
    elif ndim == 5:
      return 'NDHWC'
    else:
      raise ValueError('Input rank not supported:', ndim)
  elif data_format == 'channels_first':
    if ndim == 3:
      return 'NCW'
    elif ndim == 4:
      return 'NCHW'
    elif ndim == 5:
      return 'NCDHW'
    else:
      raise ValueError('Input rank not supported:', ndim)
  else:
    raise ValueError('Invalid data_format:', data_format)


def normalize_tuple(value, n, name):
  """Transforms a single integer or iterable of integers into an integer tuple.

  Arguments:
    value: The value to validate and convert. Could an int, or any iterable
      of ints.
    n: The size of the tuple to be returned.
    name: The name of the argument being validated, e.g. "strides" or
      "kernel_size". This is only used to format error messages.

  Returns:
    A tuple of n integers.

  Raises:
    ValueError: If something else than an int/long or iterable thereof was
      passed.
  """
  if isinstance(value, int):
    return (value,) * n
  else:
    try:
      value_tuple = tuple(value)
    except TypeError:
      raise ValueError('The `' + name + '` argument must be a tuple of ' +
                       str(n) + ' integers. Received: ' + str(value))
    if len(value_tuple) != n:
      raise ValueError('The `' + name + '` argument must be a tuple of ' +
                       str(n) + ' integers. Received: ' + str(value))
    for single_value in value_tuple:
      try:
        int(single_value)
      except ValueError:
        raise ValueError('The `' + name + '` argument must be a tuple of ' +
                         str(n) + ' integers. Received: ' + str(value) + ' '
                         'including element ' + str(single_value) + ' of type' +
                         ' ' + str(type(single_value)))
    return value_tuple


def normalize_data_format(value):
  data_format = value.lower()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('The `data_format` argument must be one of '
                     '"channels_first", "channels_last". Received: ' +
                     str(value))
  return data_format


def normalize_padding(value):
  padding = value.lower()
  if padding not in {'valid', 'same'}:
    raise ValueError('The `padding` argument must be one of "valid", "same". '
                     'Received: ' + str(padding))
  return padding


def smart_cond(pred, fn1, fn2, name=None):
  """Return either `fn1()` or `fn2()` based on the boolean predicate `pred`.

  If `pred` is a bool or has a constant value, we return either `fn1()`
  or `fn2()`, otherwise we use `tf.cond` to dynamically route to both.

  Arguments:
    pred: A scalar determining whether to return the result of `fn1` or `fn2`.
    fn1: The callable to be performed if pred is true.
    fn2: The callable to be performed if pred is false.
    name: Optional name prefix when using `tf.cond`.

  Returns:
    Tensors returned by the call to either `fn1` or `fn2`.

  Raises:
    TypeError is fn1 or fn2 is not callable.
  """
  if not callable(fn1):
    raise TypeError('`fn1` must be callable.')
  if not callable(fn2):
    raise TypeError('`fn2` must be callable.')

  pred_value = constant_value(pred)
  if pred_value is not None:
    if pred_value:
      return fn1()
    else:
      return fn2()
  else:
    return control_flow_ops.cond(pred, fn1, fn2, name)


def constant_value(pred):
  """Return the bool value for `pred`, or None if `pred` had a dynamic value.

  Arguments:
    pred: A scalar, either a Python bool or a TensorFlow boolean variable
      or tensor.

  Returns:
    True or False if `pred` has a constant boolean value, None otherwise.

  Raises:
    TypeError is pred is not a Variable, Tensor or bool.
  """
  if isinstance(pred, bool):
    pred_value = pred
  elif isinstance(pred, variables.Variable):
    pred_value = None
  elif isinstance(pred, ops.Tensor):
    pred_value = tensor_util.constant_value(pred)
  else:
    raise TypeError('`pred` must be a Tensor, a Variable, or a Python bool.')
  return pred_value
