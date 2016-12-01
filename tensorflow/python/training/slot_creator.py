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
# ==============================================================================

"""Standard functions for creating slots.

A slot is a `Variable` created with the same shape as a primary variable or
`Tensor`. A slot is always scoped in the namespace of the primary object and
typically has the same device and type.

Slots are typically used as accumulators to track values associated with
the primary object:

```python
# Optimizers can create a slot for each variable to track accumulators
accumulators = {var : create_zeros_slot(var, "momentum") for var in vs}
for var in vs:
  apply_momentum(var, accumulators[var], lr, grad, momentum_tensor)

# Slots can also be used for moving averages
mavg = create_slot(var, var.initialized_value(), "exponential_moving_avg")
update_mavg = mavg.assign_sub((mavg - var) * (1 - decay))
```
"""
# pylint: disable=g-bad-name

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables


def _create_slot_var(primary, val, scope):
  """Helper function for creating a slot variable."""

  slot = variables.Variable(val, name=scope, trainable=False)
  # pylint: disable=protected-access
  if isinstance(primary, variables.Variable) and primary._save_slice_info:
    # Primary is a partitioned variable, so we need to also indicate that
    # the slot is a partitioned variable.  Slots have the same partitioning
    # as their primaries.
    real_slot_name = scope[len(primary.op.name + "/"):-1]
    slice_info = primary._save_slice_info
    slot._set_save_slice_info(variables.Variable.SaveSliceInfo(
        slice_info.full_name + "/" + real_slot_name,
        slice_info.full_shape[:],
        slice_info.var_offset[:],
        slice_info.var_shape[:]))
  # pylint: enable=protected-access
  return slot


def create_slot(primary, val, name, colocate_with_primary=True):
  """Create a slot initialized to the given value.

  The type of the slot is determined by the given value.

  Args:
    primary: The primary `Variable` or `Tensor`.
    val: A `Tensor` specifying the initial value of the slot.
    name: Name to use for the slot variable.
    colocate_with_primary: Boolean.  If True the slot is located
      on the same device as `primary`.

  Returns:
    A `Variable` object.
  """
  # Scope the slot name in the namespace of the primary variable.
  with ops.name_scope(primary.op.name + "/" + name) as scope:
    if colocate_with_primary:
      with ops.colocate_with(primary):
        return _create_slot_var(primary, val, scope)
    else:
      return _create_slot_var(primary, val, scope)


def create_zeros_slot(primary, name, dtype=None, colocate_with_primary=True):
  """Create a slot initialized to 0 with same shape as the primary object.

  Args:
    primary: The primary `Variable` or `Tensor`.
    name: Name to use for the slot variable.
    dtype: Type of the slot variable.  Defaults to the type of `primary`.
    colocate_with_primary: Boolean.  If True the slot is located
      on the same device as `primary`.

  Returns:
    A `Variable` object.
  """
  if dtype is None:
    dtype = primary.dtype
  val = array_ops.zeros(primary.get_shape().as_list(), dtype=dtype)
  return create_slot(primary, val, name,
                     colocate_with_primary=colocate_with_primary)
