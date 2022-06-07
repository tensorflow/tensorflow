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
"""Utility to manipulate resource variables."""

from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import nest


def convert_variables_to_tensors(values):
  """Converts `ResourceVariable`s in `values` to `Tensor`s.

  If an object is a `CompositeTensor` and overrides its
  `_convert_variables_to_tensors` method, its `ResourceVariable` components
  will also be converted to `Tensor`s. Objects other than `ResourceVariable`s
  in `values` will be returned unchanged.

  Args:
    values: A nested structure of `ResourceVariable`s, or any other objects.

  Returns:
    A new structure with `ResourceVariable`s in `values` converted to `Tensor`s.
  """
  def _convert_resource_variable_to_tensor(x):
    if _pywrap_utils.IsResourceVariable(x):
      return ops.convert_to_tensor(x)
    elif isinstance(x, composite_tensor.CompositeTensor):
      return composite_tensor.convert_variables_to_tensors(x)
    else:
      return x

  return nest.map_structure(_convert_resource_variable_to_tensor, values)


def replace_variables_with_atoms(values):
  """Replaces `ResourceVariable`s in `values` with tf.nest atoms.

  This function is mostly for backward compatibility. Historically,
  `ResourceVariable`s are treated as tf.nest atoms. This is no
  longer the case after `ResourceVariable` becoming `CompositeTensor`.
  Unfortunately, tf.nest doesn't allow customization of what objects
  are treated as atoms. Calling this function to manually convert
  `ResourceVariable`s to atoms to avoid breaking tf.assert_same_structure
  with inputs of a `ResourceVariable` and an atom, like a `Tensor`.

  The specific implementation uses 0 as the tf.nest atom, but other tf.nest
  atoms could also serve the purpose. Note, the `TypeSpec` of None is not a
  tf.nest atom.

  Objects other than `ResourceVariable`s in `values` will be returned unchanged.

  Note: this function does not look into `CompositeTensor`s. Replacing
  `ResourceVariable`s in a `CompositeTensor` with atoms will change the
  `TypeSpec` of the `CompositeTensor`, which violates the semantics of
  `CompositeTensor` and tf.nest. So `ResourceVariable`s in `CompositeTensor`s
  will be returned as they are.

  Args:
    values: A nested structure of `ResourceVariable`s, or any other objects.

  Returns:
    A new structure with `ResourceVariable`s in `values` converted to atoms.
  """
  def _replace_resource_variable_with_atom(x):
    if _pywrap_utils.IsResourceVariable(x):
      return 0  # tf.nest treats 0 or tf.constant(0) as an atom.
    else:
      return x

  return nest.map_structure(_replace_resource_variable_with_atom, values)
