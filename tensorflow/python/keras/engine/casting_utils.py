# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Contains private utilities related to casting."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.mixed_precision.experimental import autocast_variable
from tensorflow.python.util import tf_contextlib


def _get_var_read_dtype(input_list):
  """Gets the dtype that AutoCastVariables should be read in."""
  try:
    # TODO(reedwm): Is choosing the first input the right choice?
    is_floating = input_list and input_list[0].dtype.is_floating
  except AttributeError:
    is_floating = False
  if is_floating:
    return input_list[0].dtype.base_dtype
  else:
    return None


@tf_contextlib.contextmanager
def autocast_context_manager(layer_weights, input_list, should_cast):
  """A context manager to autocast a layer's AutoCastVariables.

  Under this context manager, if `should_cast` is True, the AutoCastVariables in
  `layer_weights` will be casted to the dtype of the first input in
  `input_list`, if the first input is a floating-point dtype. If `should_cast`
  is False, this context manager is a no-op.

  Args:
    layer_weights: A list of weights of a layer. AutoCastVariables in this list
      will be casted if `should_cast` is True. Non-AutoCastVariables are
      ignored.
    input_list: The inputs to the layer with the AutoCastVariables.
    should_cast: Whether AutoCastVariables should be casted.

  Yields:
    Nothing.
  """
  if not should_cast:
    yield
    return

  var_read_dtype = _get_var_read_dtype(input_list)
  if var_read_dtype is None:
    yield
    return

  autocast_vars = [var for var in layer_weights
                   if isinstance(var, autocast_variable.AutoCastVariable)]
  old_read_dtypes = [var._read_dtype for var in autocast_vars]  # pylint: disable=protected-access
  for var in autocast_vars:
    var._read_dtype = var_read_dtype  # pylint: disable=protected-access
  try:
    yield
  finally:
    for var, old_read_dtype in zip(autocast_vars, old_read_dtypes):
      var._read_dtype = old_read_dtype  # pylint: disable=protected-access
