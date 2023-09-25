# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Autograph-specific overrides for ragged_tensor."""
from tensorflow.python.autograph.operators import control_flow
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops.ragged import ragged_tensor


def _tf_ragged_for_stmt(
    iter_, extra_test, body, get_state, set_state, symbol_names, opts
):
  """Overload of for_stmt that iterates over TF ragged tensors."""
  init_vars = get_state()
  control_flow.verify_loop_init_vars(init_vars, symbol_names)

  # TODO(mdan): Move this into len()? Requires eager support.
  if iter_.shape and iter_.shape[0] is not None:
    n = iter_.shape[0]
  else:
    n = iter_.row_lengths()[0]

  iterate_index = 0

  def aug_get_state():
    return (iterate_index,) + get_state()

  def aug_set_state(aug_loop_vars):
    nonlocal iterate_index
    # TODO(b/171479293): Drop the lint override.
    iterate_index, *loop_vars = aug_loop_vars  # pylint:disable=unused-variable
    # The iteration index is not "output" by the for loop. If the iteration
    # index is used outside the loop, it will appear
    # in the loop vars separately.
    set_state(loop_vars)

  def aug_body():
    nonlocal iterate_index
    body(iter_[iterate_index])
    iterate_index += 1

  def aug_test():
    main_test = iterate_index < n
    if extra_test is not None:
      return tf_cond.cond(main_test, extra_test, lambda: False)
    return main_test

  control_flow._add_max_iterations_hint(opts, n)  # pylint: disable=protected-access

  control_flow._tf_while_stmt(  # pylint: disable=protected-access
      aug_test,
      aug_body,
      aug_get_state,
      aug_set_state,
      ('<internal iterate>',) + symbol_names,
      opts,
  )


control_flow.for_loop_registry.register(
    ragged_tensor.RaggedTensor, _tf_ragged_for_stmt
)
