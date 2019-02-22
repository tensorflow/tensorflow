# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Control flow statements: loops, conditionals, etc."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.autograph.operators import special_values
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.util import nest


def for_stmt(iter_, extra_test, body, init_state):
  """Functional form of a for statement.

  The loop operates on a state, which includes all symbols that are
  variant across loop iterations, excluding the iterate as well as the
  variables local to the loop.

  For example, given the loop below that calculates the geometric and
  arithmetic means or some numbers:

    geo_mean = 1
    arith_mean = 0
    for i in range(n):
      a = numbers[i]
      geo_mean *= a
      arith_mean += a

  The state is represented by the variables geo_mean and arith_mean. The
  argument for initial_state may contain the tuple (1, 0), the body will
  include the arguments geo_mean and arith_mean and will return a tuple
  representing the new values for geo_mean and respectively arith_mean.

  Args:
    iter_: The entity being iterated over.
    extra_test: Callable with the state as arguments, and boolean return type.
        An additional loop condition.
    body: Callable with the iterate and the state as arguments, and
        state as return type. The actual loop body.
    init_state: Tuple containing the initial state.

  Returns:
    Tuple containing the final state.
  """
  if tensor_util.is_tensor(iter_):
    return _known_len_for_stmt(iter_, extra_test, body, init_state)
  elif isinstance(iter_, dataset_ops.DatasetV2):
    # Check for undefined symbols and report an error. This prevents the error
    # from propagating into the TF runtime. We have more information here and
    # can provide a clearer error message.
    undefined_symbols = _filter_undefined(init_state)

    if undefined_symbols:
      raise ValueError(
          'TensorFlow requires that the following symbols must be initialized '
          'to a Tensor, Variable or TensorArray before the loop: {}'
          .format(tuple(undefined_symbols)))

    return _dataset_for_stmt(iter_, extra_test, body, init_state)
  else:
    return _py_for_stmt(iter_, extra_test, body, init_state)


def _py_for_stmt(iter_, extra_test, body, init_state):
  """Overload of for_stmt that executes a Python for loop."""
  state = init_state
  for target in iter_:
    if extra_test is not None and not extra_test(*state):
      break
    state = body(target, *state)
  return state


def _known_len_for_stmt(iter_, extra_test, body, init_state):
  """Overload of for_stmt that iterates over objects that admit a length."""
  n = py_builtins.len_(iter_)

  def while_body(iterate_index, *state):
    iterate = iter_[iterate_index]
    new_state = body(iterate, *state)

    state = (iterate_index + 1,)
    if new_state:
      state += new_state

    return state

  def while_cond(iterate_index, *state):
    if extra_test is not None:
      return gen_math_ops.logical_and(iterate_index < n, extra_test(*state))
    return iterate_index < n

  results = while_stmt(
      while_cond,
      while_body,
      init_state=(0,) + init_state,
      extra_deps=(iter_,),
      opts=dict(maximum_iterations=n))

  # Dropping the iteration index because it's not syntactically visible.
  # TODO(mdan): Don't.
  if isinstance(results, (tuple, list)):
    assert len(results) >= 1  # Has at least the iterate.
    if len(results) > 1:
      results = results[1:]
  else:
    results = ()

  return results


def _dataset_for_stmt(ds, extra_test, body, init_state):
  """Overload of for_stmt that iterates over TF Datasets."""

  if extra_test is not None:
    raise NotImplementedError(
        'break and return statements are not yet supported in '
        'for/Dataset loops.')

  def reduce_body(state, iterate):
    new_state = body(iterate, *state)
    return new_state

  if init_state:
    return ds.reduce(init_state, reduce_body)

  # Workaround for Datset.reduce not allowing empty state tensors - create
  # a dummy state variable that remains unused.
  def reduce_body_with_dummy_state(state, iterate):
    reduce_body((), iterate)
    return state
  ds.reduce((constant_op.constant(0),), reduce_body_with_dummy_state)
  return ()


def while_stmt(test, body, init_state, extra_deps, opts=None):
  """Functional form of a while statement.

  The loop operates on a so-called state, which includes all symbols that are
  variant across loop iterations. In what follows we refer to state as either
  a tuple of entities that represent an actual state, or a list of arguments
  of the corresponding types.

  Args:
    test: Callable with the state as arguments, and boolean return type.
        The loop condition.
    body: Callable with the state as arguments, and state as return type.
        The actual loop body.
    init_state: Tuple containing the initial state.
    extra_deps: Tuple containing additional entities on which the loop may
        depend, such as loop invariants referenced by test. Used
        exclusively for dispatch control.
    opts: Optional dict of extra loop parameters.

  Returns:
    Tuple containing the final state.
  """
  # TODO(mdan): Consider adding a generic mechanism for dynamic dispatch.
  # That could be something as simple as a collection of dispatch rules, with
  # some prioritization.
  if any(tensor_util.is_tensor(v) for v in nest.flatten(extra_deps)):
    # Check for undefined symbols and report an error. This prevents the error
    # from propagating into the TF runtime. We have more information here and
    # can provide a clearer error message.
    undefined_symbols = _filter_undefined(init_state)

    if undefined_symbols:
      raise ValueError(
          'TensorFlow requires that the following symbols must be initialized '
          'to a Tensor, Variable or TensorArray before the loop: {}'
          .format(tuple(undefined_symbols)))
    return _tf_while_stmt(test, body, init_state, opts)
  else:
    return _py_while_stmt(test, body, init_state, opts)


def _filter_undefined(all_symbols):
  """Returns the names of undefined symbols contained in all_symbols."""
  undefined_symbols = [
      s.symbol_name
      for s in all_symbols
      if special_values.is_undefined(s)
  ]
  return undefined_symbols


def _tf_while_stmt(test, body, init_state, opts):
  """Overload of while_stmt that stages a TF while_stmt."""
  if opts is None:
    opts = {}

  # Non-v2 while_loop unpacks the results when there is only one return value.
  # This enforces consistency across versions.
  opts['return_same_structure'] = True

  retval = control_flow_ops.while_loop(test, body, init_state, **opts)
  return retval


def _py_while_stmt(test, body, init_state, opts):
  """Overload of while_stmt that executes a Python while loop."""
  del opts
  state = init_state
  while test(*state):
    state = body(*state)
  return state


def if_stmt(cond, body, orelse):
  """Functional form of an if statement.

  Args:
    cond: Boolean.
    body: Callable with no arguments, and outputs of the positive (if) branch
        as return type.
    orelse: Callable with no arguments, and outputs of the negative (else)
        branch as return type.

  Returns:
    Tuple containing the statement outputs.
  """
  if tensor_util.is_tensor(cond):
    return tf_if_stmt(cond, body, orelse)
  else:
    return _py_if_stmt(cond, body, orelse)


def tf_if_stmt(cond, body, orelse):
  """Overload of if_stmt that stages a TF cond."""
  protected_body = _wrap_in_protection_from_undefined(body, branch_name='if')
  protected_orelse = _wrap_in_protection_from_undefined(orelse,
                                                        branch_name='else')

  return control_flow_ops.cond(cond, protected_body, protected_orelse)


def _wrap_in_protection_from_undefined(func, branch_name):
  """Wraps function to raise useful error when it returns undefined symbols."""
  def protected_func():
    """Calls function and raises an error if undefined symbols are returned."""
    results = func()
    undefined_symbols = None
    if isinstance(results, tuple):
      undefined_symbols = _filter_undefined(results)
    elif special_values.is_undefined(results):
      # Single return value
      undefined_symbols = results.symbol_name

    if undefined_symbols:
      message = ('The following symbols must also be initialized in the %s '
                 'branch: {}. Alternatively, you may initialize them before '
                 'the if statement.') % branch_name
      message = message.format(undefined_symbols)
      raise ValueError(message)
    return results
  return protected_func


def _py_if_stmt(cond, body, orelse):
  """Overload of if_stmt that executes a Python if statement."""
  return body() if cond else orelse()
