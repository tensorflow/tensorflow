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
"""Switch case for Control Flow Operations."""

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util as util
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export

# TODO(b/269483538): needed for references while refactors are in progress
# This is to avoid a circular dependency:
# cond_v2 -> gradients_util -> control_flow_ops
cond_v2 = LazyLoader("cond_v2", globals(),
                     "tensorflow.python.ops.cond_v2")


def _indexed_case_verify_and_canonicalize_args(branch_fns, default,
                                               branch_index):
  """Verifies input arguments for the case function.

  Args:
    branch_fns: Dict or list of pairs of an `int` and a callable which returns a
      list of tensors.
    default: Optional callable that returns a list of tensors.
    branch_index: Optional int `Tensor`, which selects for the corresponding
      pred_fn_pair.

  Raises:
    TypeError: If `branch_fns` is not a list/dictionary.
    TypeError: If `branch_fns` is a list but does not contain 2-tuples or
               callables.
    TypeError: If `fns[i]` is not callable for any i, or `default` is not
               callable.

  Returns:
    branch_fns: validated list of callables for each branch (default last).
  """
  if not isinstance(branch_index, ops.Tensor):
    raise TypeError("'branch_index' must be a Tensor, got {}".format(
        type(branch_index)))
  if not branch_index.dtype.is_integer:
    raise TypeError("'branch_index' must be an integer Tensor, got {}".format(
        branch_index.dtype))

  if not branch_fns:
    raise ValueError("Must provide at least one item in 'branch_fns'")
  if not isinstance(branch_fns, (list, tuple, dict)):
    raise TypeError("'branch_fns' must be a list, tuple, or dict")

  if isinstance(branch_fns, dict):
    branch_fns = branch_fns.items()

  if all(callable(fn) for fn in branch_fns):
    branch_fns = list(enumerate(branch_fns))

  for key_fn_pair in branch_fns:
    if not isinstance(key_fn_pair, tuple) or len(key_fn_pair) != 2:
      raise TypeError("Each entry in 'branch_fns' must be a 2-tuple. "
                      f"Received {key_fn_pair}.")
    key, branch_fn = key_fn_pair

    if not isinstance(key, int):
      raise TypeError("key must be a Python `int`, got {}".format(type(key)))

    if not callable(branch_fn):
      raise TypeError("fn for key {} must be callable.".format(key))

  keys = [p[0] for p in branch_fns]
  if min(keys) < 0 or max(keys) >= len(keys) or len(set(keys)) != len(keys):
    raise ValueError(
        "branch indices (keys) must form contiguous range of [0 to {}) but "
        "found {{{}}}".format(len(keys), ",".join(map(str, sorted(keys)))))
  actions = [p[1] for p in sorted(branch_fns)]
  if default is not None:
    actions.append(default)
  return actions


def _indexed_case_helper(branch_fns,
                         default,
                         branch_index,
                         name,
                         lower_using_switch_merge=None):
  """Implementation of case that emits the n-way indexed Case op.

  Args:
    branch_fns: Dict or list of pairs of a boolean scalar tensor, and a callable
      which returns a list of tensors.
    default: Optional callable that returns a list of tensors.
    branch_index: Optional int `Tensor`, which selects for the corresponding
      pred_fn_pair.
    name: A name for this operation (optional).
    lower_using_switch_merge: Lower this op using switch merge ops (optional).

  Returns:
    The tensors returned by the pair whose key matched branch_index, or
    those returned by `default` if none does.

  Raises:
    TypeError: If `branch_fns` is not a list/dictionary.
    TypeError: If `branch_fns` is a list but does not contain 2-tuples or
               callables.
    TypeError: If `fns[i]` is not callable for any i, or `default` is not
               callable.
  """
  branch_fns = _indexed_case_verify_and_canonicalize_args(
      branch_fns, default, branch_index)
  with ops.name_scope(name, "case", [branch_index]):
    if context.executing_eagerly() and not hasattr(branch_index, "graph"):
      branch_index = array_ops.where(
          math_ops.less(branch_index, 0)
          | math_ops.greater_equal(branch_index, len(branch_fns)),
          len(branch_fns) - 1, branch_index)
      return branch_fns[int(branch_index)]()
    return cond_v2.indexed_case(
        branch_index,
        branch_fns,
        lower_using_switch_merge=lower_using_switch_merge)


@tf_export("__internal__.execute_fn_for_device", v1=[])
def execute_fn_for_device(device_branch_fns, default_fn, name="execute_fn"):
  """Executes one of the provided callables based on the device placement.

  This API is used when the implementations for high level function depend on
  the underlying device placement. It takes a dictionary of device type to
  callables. The device type includes "CPU", "GPU", "TPU", etc. When the type of
  the device where to run this op matches the key in 'device_branch_fns',
  the corresponding callable is executed, falling back to 'default_fn' if none
  matches.

  **Example:**
  ```python
  def f1(): return tf.constant(1)
  def f2(): return tf.constant(2)
  r = tf.execute_fn_for_device({"CPU": f1, "GPU": f2}, default_fn=f1)
  ```
  'r' is evaluated as 1 when it runs on CPU, 2 running on GPU, 1 running on
  any other device types.


  Args:
    device_branch_fns: a dictionary of device types to the callables. Each
      callable must return a matching structure of tensors.
    default_fn: fallback callable when the underlying device does not match any
      key in the 'device_branch_fns'.
    name: A name for this operation (optional).

  Returns:
    The tensors returned by the callable identified by device type during
    execution, or those returned by 'default_fn' if no key matches.
  """
  # Always execute the default fn for XLA to avoid complicated graph by case op.
  # see more discussions in b/167276293.
  is_in_xla = util.GraphOrParentsInXlaContext(ops.get_default_graph())
  if is_in_xla:
    return default_fn()
  device_branch_fns_upper = {k.upper(): v for k, v in device_branch_fns.items()}
  branch_fns = list(device_branch_fns_upper.values())
  devices = list(device_branch_fns_upper.keys())
  device_index = gen_functional_ops.device_index(device_names=devices)
  return _indexed_case_helper(
      branch_fns,
      default_fn,
      device_index,
      name,
      lower_using_switch_merge=False)


@tf_export("switch_case")
def switch_case(branch_index, branch_fns, default=None, name="switch_case"):
  """Create a switch/case operation, i.e.

  an integer-indexed conditional.

  See also `tf.case`.

  This op can be substantially more efficient than `tf.case` when exactly one
  branch will be selected. `tf.switch_case` is more like a C++ switch/case
  statement than `tf.case`, which is more like an if/elif/elif/else chain.

  The `branch_fns` parameter is either a dict from `int` to callables, or list
  of (`int`, callable) pairs, or simply a list of callables (in which case the
  index is implicitly the key). The `branch_index` `Tensor` is used to select an
  element in `branch_fns` with matching `int` key, falling back to `default`
  if none match, or `max(keys)` if no `default` is provided. The keys must form
  a contiguous set from `0` to `len(branch_fns) - 1`.

  `tf.switch_case` supports nested structures as implemented in `tf.nest`. All
  callables must return the same (possibly nested) value structure of lists,
  tuples, and/or named tuples.

  **Example:**

  Pseudocode:

  ```c++
  switch (branch_index) {  // c-style switch
    case 0: return 17;
    case 1: return 31;
    default: return -1;
  }
  ```
  or
  ```python
  branches = {0: lambda: 17, 1: lambda: 31}
  branches.get(branch_index, lambda: -1)()
  ```

  Expressions:

  ```python
  def f1(): return tf.constant(17)
  def f2(): return tf.constant(31)
  def f3(): return tf.constant(-1)
  r = tf.switch_case(branch_index, branch_fns={0: f1, 1: f2}, default=f3)
  # Equivalent: tf.switch_case(branch_index, branch_fns={0: f1, 1: f2, 2: f3})
  ```

  Args:
    branch_index: An int Tensor specifying which of `branch_fns` should be
      executed.
    branch_fns: A `dict` mapping `int`s to callables, or a `list` of (`int`,
      callable) pairs, or simply a list of callables (in which case the index
      serves as the key). Each callable must return a matching structure of
      tensors.
    default: Optional callable that returns a structure of tensors.
    name: A name for this operation (optional).

  Returns:
    The tensors returned by the callable identified by `branch_index`, or those
    returned by `default` if no key matches and `default` was provided, or those
    returned by the max-keyed `branch_fn` if no `default` is provided.

  Raises:
    TypeError: If `branch_fns` is not a list/dictionary.
    TypeError: If `branch_fns` is a list but does not contain 2-tuples or
               callables.
    TypeError: If `fns[i]` is not callable for any i, or `default` is not
               callable.
  """
  return _indexed_case_helper(branch_fns, default, branch_index, name)
