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


def stack(list_or_tensor, element_dtype=None):
  """Stacks the input, if it admits the notion of stacking. No-op otherwise.

  For example, a list of tensors can be stacked into a larger tensor. This
  function is similar to tf.stack, but it accepts non-lists and lists of
  non-tensors as arguments. In the latter case, the function does nothing.

  Args:
    list_or_tensor: Any entity.
    element_dtype: Optional dtype for the elements in the list. Required if the
        input is stackable, and the list is untyped.

  Returns:
    If the input is stackable, a new object representing the stacked inputs.
  Otherwise it returns list_or_tensor unchanged.
  """
  return data_structures.list_stack(
      list_or_tensor,
      data_structures.ListStackOpts(
          element_dtype=element_dtype, original_call=lambda x: x))
