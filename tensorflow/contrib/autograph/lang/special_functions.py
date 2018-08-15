# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Special functions that only make sense for AutoGraph.

These functions are meant to ensure feature parity between Python and AutoGraph,
so that the exact same code works in both modes. In general, AutoGraph will
replace these calls.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.autograph.operators import data_structures


def tensor_list(elements,
                element_dtype=None,
                element_shape=None,
                use_tensor_array=False):
  """Creates an tensor list and populates it with the given elements.

  This function provides a more uniform access to tensor lists and tensor
  arrays, and allows optional initialization.

  Note: this function is a simplified wrapper. If you need greater control,
  it is recommended to use the underlying implementation directly.

  Args:
    elements: Iterable[tf.Tensor, ...], the elements to initially fill the list
        with
    element_dtype: Optional[tf.DType], data type for the elements in the list;
        required if the list is empty
    element_shape: Optional[tf.TensorShape], shape for the elements in the list;
        required if the list is empty
    use_tensor_array: bool, whether to use the more compatible but restrictive
        tf.TensorArray implementation
  Returns:
    Union[tf.Tensor, tf.TensorArray], the new list.
  Raises:
    ValueError: for invalid arguments
  """
  if not (elements or (element_dtype and element_shape)):
    raise ValueError(
        'element_dtype and element_shape are required for empty lists')
  if use_tensor_array:
    return data_structures.tf_tensor_array_new(elements, element_dtype,
                                               element_shape)
  else:
    return data_structures.tf_tensor_list_new(elements, element_dtype,
                                              element_shape)


def stack(list_or_tensor, element_dtype=None, strict=True):
  """Stacks the input, if it admits the notion of stacking.

  For example, a list of tensors can be stacked into a larger tensor. This
  function is similar to tf.stack, but it accepts non-lists and lists of
  non-tensors as arguments. In the latter case, the function does nothing.

  Args:
    list_or_tensor: Any
    element_dtype: tf.DType, optional dtypedtype for the elements in the list.
        Required if the input is stackable, and the list is untyped.
    strict: bool, if True an error is raised if the input is not stackable.
        Otherwise the function is a no-op.

  Returns:
    Any, if the input is stackable, the result will be a tf.Tensor. Otherwise,
    if strict=False, the result will be list_or_tensor.

  Raises:
    ValueError: if strict=True and the input is not stackable.
  """
  if strict:
    def raise_error(x):
      raise ValueError('%s must be stackable when strict=True' % x)
    original_call = raise_error
  else:
    original_call = lambda x: x
  return data_structures.list_stack(
      list_or_tensor,
      data_structures.ListStackOpts(
          element_dtype=element_dtype, original_call=original_call))
