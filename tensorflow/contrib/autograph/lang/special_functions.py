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
