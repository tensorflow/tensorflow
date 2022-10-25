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
"""Utility to manipulate CompositeTensors in tf.function."""

from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import nest


# TODO(b/240337581, b/240337099): Remove this function when we de-alias
# dt_resource tensors or tf.nest support is_leaf.
def flatten_with_variables(inputs):
  """Flattens `inputs` but don't expand `ResourceVariable`s."""
  # We assume that any CompositeTensors have already converted their components
  # from numpy arrays to Tensors, so we don't need to expand composites here for
  # the numpy array conversion. Instead, we do so because the flattened inputs
  # are eventually passed to ConcreteFunction()._call_flat, which requires
  # expanded composites.
  flat_inputs = []
  for value in nest.flatten(inputs):
    if (isinstance(value, composite_tensor.CompositeTensor) and
        not _pywrap_utils.IsResourceVariable(value)):
      components = value._type_spec._to_components(value)  # pylint: disable=protected-access
      flat_inputs.extend(flatten_with_variables(components))
    else:
      flat_inputs.append(value)
  return flat_inputs


# TODO(b/246437883): Consider removing this helper function once the variable
# branch is removed from _get_defun_input.
def flatten_with_variables_or_variable_specs(arg):
  """Gets defun input and doesn't expand `ResourceVariable`s or `VariableSpec`s."""
  flat_inputs = []
  for value in nest.flatten(arg):
    if (isinstance(value, composite_tensor.CompositeTensor) and
        not _pywrap_utils.IsResourceVariable(value)):
      # Replace any composite tensors with their TypeSpecs. This is important
      # for ensuring that shape information that's not preserved by the
      # TypeSpec (such as the number of values in a SparseTensor) gets
      # properly masked.
      spec = value._type_spec  # pylint: disable=protected-access
      flat_inputs.extend(flatten_with_variables_or_variable_specs(spec))
    elif (isinstance(value, type_spec.TypeSpec) and
          not isinstance(value, tensor_spec.TensorSpec) and
          not isinstance(value, resource_variable_ops.VariableSpec)):
      component_specs = value._component_specs  # pylint: disable=protected-access
      components = flatten_with_variables_or_variable_specs(component_specs)
      flat_inputs.extend(components)
    else:
      flat_inputs.append(value)
  return flat_inputs
