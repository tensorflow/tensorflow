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
from tensorflow.python.autograph.utils import ag_logging
from tensorflow.python.data.experimental.ops import scan_ops
from tensorflow.python.data.experimental.ops import take_while_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import tensor_array_ops


LIMIT_PYTHON_ITERATIONS = True
PYTHON_MAX_ITERATIONS = 100000000  # Fails in about one minute for empty loops.
WARN_INEFFICIENT_UNROLL = True
INEFFICIENT_UNROLL_MIN_ITERATIONS = 3000
INEFFICIENT_UNROLL_MIN_OPS = 1


def _disallow_undefs_into_loop(*values):
  """Ensures that all values in the state are defined when entering a loop."""
  undefined = tuple(filter(special_values.is_undefined, values))
  if undefined:
    raise ValueError(
        'TensorFlow requires that the following symbols must be defined'
        ' before the loop: {}'.format(
            tuple(s.symbol_name for s in undefined)))

  for value in values:
    if special_values.is_undefined_return(value):
      # Assumption: the loop will only capture the variable which tracks the
      # return value if the loop contained a return statement.
      # TODO(mdan): This should be checked at the place where return occurs.
      raise ValueError(
          'Return statements are not supported within a TensorFlow loop.')


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
    return _known_len_tf_for_stmt(iter_, extra_test, body, init_state)

  if isinstance(iter_, dataset_ops.DatasetV2):
    return _tf_dataset_for_stmt(iter_, extra_test, body, init_state)

  if isinstance(iter_, iterator_ops.IteratorV2):
    return _tf_iterator_for_stmt(iter_, extra_test, body, init_state)

  # Note: This experimental interface is subject to change.
  custom_handler = getattr(iter_, '_autograph_for_loop', None)
  if custom_handler is not None:
    # TODO(mdan): TensorFlow-specific verification - handlers should perform it.
    _disallow_undefs_into_loop(*init_state)
    return custom_handler(extra_test, body, init_state)

  return _py_for_stmt(iter_, extra_test, body, init_state)


def _py_for_stmt(iter_, extra_test, body, init_state):
  """Overload of for_stmt that executes a Python for loop."""
  state = init_state
  for target in iter_:
    if extra_test is not None and not extra_test(*state):
      break
    state = body(target, *state)
  return state


def _known_len_tf_for_stmt(iter_, extra_test, body, init_state):
  """Overload of for_stmt that iterates over TF entities that admit a length."""
  _disallow_undefs_into_loop(*init_state)

  n = py_builtins.len_(iter_)
  # TODO(b/117628877): Revisit performance once XLA has the necessary support.
  # Note: using a TensorArray creates an extra copy, but can calculate
  # gradients more efficiently than StridedSlice.
  ta = tensor_array_ops.TensorArray(iter_.dtype, size=n)
  iter_ = ta.unstack(iter_)

  def while_body(iterate_index, *state):
    iterate = iter_.read(iterate_index)
    new_state = body(iterate, *state)

    state = (iterate_index + 1,)
    if new_state:
      state += new_state

    return state

  def while_cond(iterate_index, *state):
    if extra_test is not None:
      return control_flow_ops.cond(
          iterate_index < n,
          lambda: extra_test(*state),
          lambda: False)
    return iterate_index < n

  results = _tf_while_stmt(
      while_cond,
      while_body,
      init_state=(0,) + init_state,
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


def _tf_iterator_for_stmt(itr, extra_test, body, init_state):
  """Overload of for_stmt that iterates over TF Iterators. See for_loop."""
  _disallow_undefs_into_loop(*init_state)

  def while_body_actual(opt_iterate, *state):
    new_state = body(opt_iterate.get_value(), *state)
    # TODO(mdan): Fix this inconsistency in the converter.
    if new_state is None:
      new_state = ()
    return new_state

  def while_body(has_next, state):
    """Main loop body."""
    opt_iterate = iterator_ops.get_next_as_optional(itr)
    has_next = opt_iterate.has_value()

    if not init_state:
      # cond_v2 requires at least one state tensor in V1.
      dummy_state = (constant_op.constant(()),)
    else:
      dummy_state = ()

    # TODO(mdan): If tf.while_loop supported Optional, this could be avoided.
    new_state = control_flow_ops.cond(
        has_next,
        lambda: dummy_state + while_body_actual(opt_iterate, *state),
        lambda: dummy_state + state)

    if dummy_state:
      new_state = new_state[1:]

    return has_next, new_state

  def while_cond(has_next, state):
    if extra_test is not None:
      return control_flow_ops.cond(
          has_next,
          lambda: extra_test(*state),
          lambda: False)
    return has_next

  _, final_state = _tf_while_stmt(
      while_cond,
      while_body,
      init_state=(True, init_state),
      opts=None)
  return final_state


def _tf_dataset_for_stmt(ds, extra_test, body, init_state):
  """Overload of for_stmt that iterates over TF Datasets."""
  _disallow_undefs_into_loop(*init_state)

  if extra_test is not None:
    assert init_state, 'Lowering should always add state.'
    return _dataset_for_stmt_with_extra_test(ds, extra_test, body, init_state)

  return _dataset_for_stmt_no_extra_test(ds, body, init_state)


def _dataset_for_stmt_with_extra_test(ds, extra_test, body, init_state):
  """Overload of _dataset_for_stmt with early stopping. See for_stmt."""

  def scan_body(state, iterate):
    extra_cond = extra_test(*state)
    new_state = control_flow_ops.cond(
        extra_cond, lambda: body(iterate, *state), lambda: state)
    aug_state = new_state, extra_cond
    # Note: new_state is the actual state of scan; aug_state is its output
    # (hence the redundancy).
    return new_state, aug_state

  def take_while_predicate(new_state, extra_cond):
    del new_state
    return extra_cond

  def reduce_body(old_state, aug_state):
    del old_state
    new_state, extra_cond = aug_state
    del extra_cond
    return new_state

  ds = ds.apply(scan_ops.scan(init_state, scan_body))
  ds = ds.apply(take_while_ops.take_while(take_while_predicate))
  return ds.reduce(init_state, reduce_body)


def _dataset_for_stmt_no_extra_test(ds, body, init_state):
  """Overload of _dataset_for_stmt without early stopping. See for_stmt."""

  def reduce_body(state, iterate):
    new_state = body(iterate, *state)
    return new_state

  if init_state:
    return ds.reduce(init_state, reduce_body)

  # Workaround for Dataset.reduce not allowing empty state tensors - create
  # a dummy state variable that remains unused.
  def reduce_body_with_dummy_state(state, iterate):
    reduce_body((), iterate)
    return state
  ds.reduce((constant_op.constant(0),), reduce_body_with_dummy_state)
  return ()


def while_stmt(test, body, init_state, opts=None):
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
    opts: Optional dict of extra loop parameters.

  Returns:
    Tuple containing the final state.
  """
  # Evaluate the initial test once in order to do the dispatch. The evaluation
  # is isolated to minimize unwanted side effects.
  # TODO(mdan): Do a full iteration - some state types might lower to Tensor.
  with func_graph.FuncGraph('tmp').as_default():
    init_test = test(*init_state)

  # TensorFlow: Multiple evaluations are acceptable in this case, so we're fine
  # with the re-evaluation of `test` that `_tf_while_stmt` will make.
  if tensor_util.is_tensor(init_test):
    return _tf_while_stmt(test, body, init_state, opts)

  # Normal Python: We already consumed one evaluation of `test`; consistently,
  # unroll one iteration before dispatching to a normal loop.
  # TODO(mdan): Push the "init_test" value via opts into _py_while_stmt?
  if not init_test:
    return init_state
  init_state = body(*init_state)

  return _py_while_stmt(test, body, init_state, opts)


def _tf_while_stmt(test, body, init_state, opts):
  """Overload of while_stmt that stages a TF while_stmt."""
  _disallow_undefs_into_loop(*init_state)

  if opts is None:
    opts = {}

  # Non-v2 while_loop unpacks the results when there is only one return value.
  # This enforces consistency across versions.
  opts['return_same_structure'] = True

  retval = control_flow_ops.while_loop(test, body, init_state, **opts)
  return retval


class _PythonLoopChecker(object):
  """Verifies Python loops for TF-specific limits."""

  def __init__(self):
    self.iterations = 0
    self.check_inefficient_unroll = WARN_INEFFICIENT_UNROLL

    # Triggered when we decided to test the op counts.
    self.check_op_count_after_iteration = False

  def _get_ops(self):
    return ops.get_default_graph().get_operations()

  def _check_unroll_limits(self):
    if LIMIT_PYTHON_ITERATIONS and self.iterations > PYTHON_MAX_ITERATIONS:
      raise ValueError('iteration limit exceeded')

  def _stop_checking_inefficient_unroll(self):
    self.check_inefficient_unroll = False
    self.ops_before_iteration = None

  def _verify_ineffcient_unroll(self):
    """Checks for possibly-inefficient creation of ops in a Python loop."""
    assert self.ops_before_iteration is not None
    ops_after_iteration = self._get_ops()
    new_ops = tuple(
        op for op in ops_after_iteration if op not in self.ops_before_iteration)

    if len(new_ops) < INEFFICIENT_UNROLL_MIN_OPS:
      return False

    # TODO(mdan): Add location information.
    ag_logging.warn(
        'TensorFlow ops are being created in a Python loop with large number'
        ' of iterations. This can lead to slow startup. Did you mean to use a'
        ' TensorFlow loop? For example, `while True:` is a Python loop, and'
        ' `while tf.constant(True):` is a TensorFlow loop. The following'
        ' ops were created after iteration %s: %s', self.iterations, new_ops)
    return True

  def before_iteration(self):
    """Called before each iteration in a Python loop."""
    if (self.check_inefficient_unroll and
        self.iterations > INEFFICIENT_UNROLL_MIN_ITERATIONS):
      self.ops_before_iteration = self._get_ops()
      self.check_op_count_after_iteration = True

  def after_iteration(self):
    """Called after each iteration in a Python loop."""
    self.iterations += 1

    self._check_unroll_limits()

    if self.check_inefficient_unroll and self.check_op_count_after_iteration:
      did_warn = self._verify_ineffcient_unroll()
      if did_warn:
        self._stop_checking_inefficient_unroll()  # Only warn once.
      elif self.iterations > INEFFICIENT_UNROLL_MIN_ITERATIONS + 3:
        # Once deciding to check the op counts, only do it for a few iterations.
        self._stop_checking_inefficient_unroll()


def _py_while_stmt(test, body, init_state, opts):
  """Overload of while_stmt that executes a Python while loop."""
  del opts

  if __debug__:
    checker = _PythonLoopChecker()

  state = init_state
  while test(*state):

    if __debug__:
      checker.before_iteration()

    state = body(*state)

    if __debug__:
      checker.after_iteration()

  return state


def if_stmt(cond, body, orelse, get_state, set_state):
  """Functional form of an if statement.

  Args:
    cond: Boolean.
    body: Callable with no arguments, and outputs of the positive (if) branch
        as return type.
    orelse: Callable with no arguments, and outputs of the negative (else)
        branch as return type.
    get_state: Function that returns a tuple containing the values of all
        composite symbols modified within the conditional. This allows access to
        state that branches may mutate through side effects. This function is
        not needed and should not be called when dispatching to code matching
        Python's default semantics. This is useful for checkpointing to avoid
        unintended side-effects when staging requires evaluating all code-paths.
    set_state: Function to set the values of all composite symbols modified
        within the conditional. This is the complement to get_state, used to
        restore checkpointed values. The single argument a tuple containing
        values for each composite symbol that may be modified in a branch of the
        conditional. The is usually the result of a call to get_state.

  Returns:
    Tuple containing the statement outputs.
  """
  if tensor_util.is_tensor(cond):
    return tf_if_stmt(cond, body, orelse, get_state, set_state)
  else:
    return _py_if_stmt(cond, body, orelse)


def tf_if_stmt(cond, body, orelse, get_state, set_state):
  """Overload of if_stmt that stages a TF cond."""
  body = _wrap_disallow_undefs_from_cond(body, branch_name='if')
  orelse = _wrap_disallow_undefs_from_cond(orelse, branch_name='else')
  body = _isolate_state(body, get_state, set_state)
  orelse = _isolate_state(orelse, get_state, set_state)

  # `state` currently includes the values of any composite symbols (e.g. `a.b`)
  # composites modified by the loop. `outputs` includes the values of basic
  # symbols (e.g. `a`) which cannot be passed by reference and must be returned.
  # See _isolate_state.
  # TODO(mdan): We should minimize calls to get/set_state.
  outputs, final_state = control_flow_ops.cond(cond, body, orelse)
  set_state(final_state)

  return outputs


def _isolate_state(func, get_state, set_state):
  """Wraps func to (best-effort) isolate state mutations that func may do.

  The simplest example of state mutation is mutation of variables (via e.g.
  attributes), or modification of globals.

  This allows us to more safely execute this function without worrying about
  side effects when the function wasn't normally expected to execute. For
  example, staging requires that the function is executed ahead of time, and
  we need to ensure its effects are not observed during normal execution.

  Args:
    func: () -> Any
    get_state: () -> Any, returns the current state
    set_state: (Any) -> None, resets the state to the specified values.
      Typically the result of an earlier call to `get_state`.

  Returns:
    Tuple[Any, Any], where the first element is the return value of `func`,
    and the second is the final state values.
  """

  def wrapper():
    init_state = get_state()
    outputs = func()
    # TODO(mdan): These should be copies, lest set_state might affect them.
    final_state = get_state()
    set_state(init_state)
    return outputs, final_state

  return wrapper


def _wrap_disallow_undefs_from_cond(func, branch_name):
  """Wraps conditional branch to disallow returning undefined symbols."""

  def wrapper():
    """Calls function and raises an error if undefined symbols are returned."""
    results = func()

    if isinstance(results, tuple):
      results_tuple = results
    else:
      results_tuple = results,
    undefined = tuple(filter(special_values.is_undefined, results_tuple))
    if undefined:
      raise ValueError(
          'The following symbols must also be initialized in the {} branch: {}.'
          ' Alternatively, you may initialize them before the if'
          ' statement.'.format(branch_name,
                               tuple(s.symbol_name for s in undefined)))

    for result in results_tuple:
      if special_values.is_undefined_return(result):
        raise ValueError(
            'A value must also be returned from the {} branch. If a value is '
            'returned from one branch of a conditional a value must be '
            'returned from all branches.'.format(branch_name))

    return results

  return wrapper


def _py_if_stmt(cond, body, orelse):
  """Overload of if_stmt that executes a Python if statement."""
  return body() if cond else orelse()
