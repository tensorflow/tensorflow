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
"""Control flow statements: loops, conditionals, etc.

Note: most of these operators accept pairs of get_state/set_state functions, to
capture mutations that the corresponding code blocks might make. These
mutations only need to be captured when staging the control flow, and they just
work when reverting to Python behavior.

__Examples__

```
while cond:
  self.x += i
```

When the functionalized version is executed as a Python loop, it just works:

```
def loop_body():
  self.x += i     # works as expected for Python loops
```

But it won't work for TF loops:

```
def loop_body():
  self.x += i     # self.x has the wrong value!
```

get_state/set_state allow piping the mutations through the loop variables as
well, in effect changing the loop body:

```
def loop_body(self_x):
  self.x = self_x  # self.x now has the proper value
  self.x += i      # the original block
  self_x = self.x  # write self.x back into the loop vars
  return self_x

self_x = tf.while_loop(...)
self.x = self_x    # the result is not properly captured
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import traceback

import numpy as np

from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.autograph.operators import special_values
from tensorflow.python.autograph.utils import ag_logging
from tensorflow.python.autograph.utils import compat_util
from tensorflow.python.autograph.utils import misc
from tensorflow.python.autograph.utils import tensors
from tensorflow.python.data.experimental.ops import scan_ops
from tensorflow.python.data.experimental.ops import take_while_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import lazy_loader
from tensorflow.python.util import nest


# TODO(b/145618471): Remove this dependency.
# Lazy import to work around circular dependencies
input_lib = lazy_loader.LazyLoader(
    'input_lib', globals(),
    'tensorflow.python.distribute.input_lib')

PYTHON_MAX_ITERATIONS = 100000000  # Fails in about one minute for empty loops.
WARN_INEFFICIENT_UNROLL = True
INEFFICIENT_UNROLL_MIN_ITERATIONS = 3000
INEFFICIENT_UNROLL_MIN_OPS = 1


# TODO(mdan): Use the custom operator pattern instead of type dispatch.
# An example of this pattern is found in the implementation of distributed
# datasets. Before it can be used though, we need to standardize the interface.


def _verify_loop_init_vars(values, symbol_names):
  """Ensures that all values in the state are defined when entering a loop."""
  for name, value in zip(symbol_names, values):
    if value is None:
      raise ValueError('"{}" may not be None before the loop.'.format(name))
    if special_values.is_undefined_return(value):
      # Assumption: the loop will only capture the variable which tracks the
      # return value if the loop contained a return statement.
      # TODO(mdan): This should be checked at the place where return occurs.
      raise ValueError(
          'return statements are not supported within a TensorFlow loop.')
    if special_values.is_undefined(value):
      raise ValueError('"{}" must be defined before the loop.'.format(name))


def _is_subshape(left, right):
  """Returns True if left shape is at least as specific as right shape."""
  # TODO(mdan): This code should be in TensorShape.
  # Note: this is not the same as TensorShape.is_compatible_with, which is
  # symmetric.
  # This code also duplicates _ShapeLessThanOrEqual from  control_flow_ops.py.
  if right.dims is None:
    return True
  if left.ndims != right.ndims:
    return False
  for ldim, rdim in zip(left.dims, right.dims):
    if rdim.value is not None and ldim.value != rdim.value:
      return False
  return True


# TODO(mdan): Remove these verifications once TF ops can properly report names.
def _verify_single_loop_var(
    name, check_shape, init, entry, exit_, shape_invariant):
  """Verifies whether the initial, entry and exit values are consistent."""
  assert entry is not None, 'no TF op should set "{}" to None?'.format(name)
  if exit_ is None:
    raise ValueError('"{}" is None at the end of the iteration.'.format(name))

  if isinstance(init, (bool, int, float, str, np.ndarray)):
    init = ops.convert_to_tensor_v2(init)
  if isinstance(entry, (bool, int, float, str, np.ndarray)):
    entry = ops.convert_to_tensor_v2(entry)
  if isinstance(exit_, (bool, int, float, str, np.ndarray)):
    exit_ = ops.convert_to_tensor_v2(exit_)

  if (not tensor_util.is_tensor(entry) or
      not tensor_util.is_tensor(exit_)):
    return

  # TODO(mdan): Properly account for CompositeTensors.
  if (not hasattr(entry, 'dtype') or
      not hasattr(exit_, 'dtype')):
    return
  if (not hasattr(entry, 'shape') or
      not hasattr(exit_, 'shape')):
    return

  if entry.dtype != exit_.dtype:
    raise TypeError(
        '"{}" has dtype {} before the loop, but dtype {} after one'
        ' iteration. TensorFlow control flow requires it stays the'
        ' same.'.format(
            name,
            entry.dtype.name,
            exit_.dtype.name,
        ))
  if check_shape:
    exit_shape = exit_.shape
    if shape_invariant is None:
      entry_shape = entry.shape
      if not _is_subshape(exit_shape, entry_shape):
        raise ValueError(
            '"{}" has shape {} before the loop, but shape {} after one'
            ' iteration. Use tf.autograph.experimental.set_loop_options to set'
            ' shape invariants.'.format(name, entry_shape, exit_shape))
    else:
      init_shape = init.shape
      if not _is_subshape(init_shape, shape_invariant):
        raise ValueError(
            '"{}" has shape {} before the loop, which does not conform with'
            ' the shape invariant {}.'.format(name, init_shape,
                                              shape_invariant))
      if not _is_subshape(exit_shape, shape_invariant):
        raise ValueError(
            '"{}" has shape {} after one iteration, which does not conform with'
            ' the shape invariant {}.'.format(
                name, exit_shape, shape_invariant))


def _verify_tf_loop_vars(init_vars,
                         iter_entry_vars,
                         iter_exit_vars,
                         symbol_names,
                         opts,
                         check_shapes=True):
  """Verifies loop variables for consistency."""
  if check_shapes and 'shape_invariants' in opts:
    shape_invariants = opts['shape_invariants']
  else:
    shape_invariants = nest.map_structure(lambda _: None, iter_entry_vars)

  assert len(symbol_names) == len(shape_invariants)
  assert len(symbol_names) == len(init_vars)
  assert len(symbol_names) == len(iter_entry_vars)
  assert len(symbol_names) == len(iter_exit_vars)

  for i in range(len(symbol_names)):
    name = symbol_names[i]
    init = init_vars[i]
    entry = iter_entry_vars[i]
    exit_ = iter_exit_vars[i]
    invariant = shape_invariants[i]

    try:
      nest.assert_same_structure(init, entry, expand_composites=True)
      nest.assert_same_structure(entry, exit_, expand_composites=True)
    except (ValueError, TypeError) as e:
      raise TypeError('"{}" does not have the same nested structure after one'
                      ' iteration.\n\n{}'.format(name, e))
    if invariant is not None:
      try:
        nest.assert_same_structure(init, invariant, expand_composites=False)
      except (ValueError, TypeError) as e:
        raise TypeError('"{}" does not have the same nested structure as its'
                        ' corresponding shape invariant.\n\n{}'.format(name, e))

    nest.map_structure(
        functools.partial(_verify_single_loop_var, name, check_shapes), init,
        entry, exit_, invariant)


def _verify_single_cond_var(name, body_var, orelse_var):
  """Verifies whether body_var and orelse_var are consistent."""
  if body_var is None:
    raise ValueError('"{}" is None at the end of the TRUE branch.'.format(name))
  if orelse_var is None:
    raise ValueError(
        '"{}" is None at the end of the FALSE branch.'.format(name))

  if isinstance(body_var, (bool, int, float, str, np.ndarray)):
    body_var = ops.convert_to_tensor_v2(body_var)

  if isinstance(orelse_var, (bool, int, float, str, np.ndarray)):
    orelse_var = ops.convert_to_tensor_v2(orelse_var)

  if (not tensor_util.is_tensor(body_var) or
      not tensor_util.is_tensor(orelse_var)):
    return

  # TODO(mdan): Properly account for CompositeTensors.
  if (not hasattr(body_var, 'dtype') or
      not hasattr(orelse_var, 'dtype')):
    return

  if body_var.dtype != orelse_var.dtype:
    raise TypeError(
        '"{}" has dtype {} in the TRUE branch, but dtype={} in the FALSE'
        ' branch. TensorFlow control flow requires that they are the'
        ' same.'.format(name, body_var.dtype.name,
                        orelse_var.dtype.name))


def _verify_tf_cond_vars(body_vars, orelse_vars, symbol_names):
  """Verifies variables manipulated by a conditional for consistency."""
  basic_body_vars, composite_body_vars = body_vars
  basic_orelse_vars, composite_orelse_vars = orelse_vars
  assert isinstance(composite_body_vars, tuple)
  assert isinstance(composite_orelse_vars, tuple)

  # TODO(kkb): Make this more consistent.
  # The basic outputs should always be a tuple.
  if not isinstance(basic_body_vars, tuple):
    basic_body_vars = (basic_body_vars,)
  if not isinstance(basic_orelse_vars, tuple):
    basic_orelse_vars = (basic_orelse_vars,)

  body_vars = basic_body_vars + composite_body_vars
  orelse_vars = basic_orelse_vars + composite_orelse_vars

  named_vars = zip(symbol_names, body_vars, orelse_vars)
  for name, body_var, orelse_var in named_vars:
    try:
      nest.assert_same_structure(
          body_var, orelse_var, expand_composites=True)
    except (ValueError, TypeError) as e:
      raise TypeError(
          '"{}" does not have the same nested structure in the TRUE and FALSE'
          ' branches.\n\n{}'.format(name, str(e)))

    nest.map_structure(
        functools.partial(_verify_single_cond_var, name), body_var, orelse_var)


def for_stmt(iter_, extra_test, body, get_state, set_state, symbol_names, opts):
  """Functional form of a for statement.

  The loop operates on a state, which includes all symbols that are
  variant across loop iterations, excluding the variables local to the loop.

  For example, given the loop below that calculates the geometric and
  arithmetic means or some numbers:

  ```
    geo_mean = 1
    arith_mean = 0
    for i in range(n):
      a = numbers[i]
      geo_mean *= a
      arith_mean += a
  ```

  The state is represented by the variables geo_mean and arith_mean. The
  `extra_test`, `body`, `get_state` and `set_state` functions must bind to the
  original `geo_mean` and `arith_mean` symbols, using `nonlocal`.

  Args:
    iter_: The entity being iterated over.
    extra_test: Callable with the state as arguments, and boolean return type.
      An additional loop condition.
    body: Callable with the iterate and the state as arguments, and state as
      return type. The actual loop body.
    get_state: Additional callable which can capture additional state (such as
      the values of composite symbols). This is only useful when staging the
      loop.
    set_state: Additional callable which save values captured by get_state back
      into the Python environment. This is only useful when staging the loop.
    symbol_names: Tuple containing names of the loop variables returned by
      get_state.
    opts: Optional dict of extra loop parameters.

  Returns:
    Tuple containing the final state.
  """
  if tensor_util.is_tensor(iter_):
    if tensors.is_range_tensor(iter_):
      _tf_range_for_stmt(
          iter_, extra_test, body, get_state, set_state, symbol_names, opts)
    else:
      _known_len_tf_for_stmt(
          iter_, extra_test, body, get_state, set_state, symbol_names, opts)

  elif isinstance(iter_, dataset_ops.DatasetV2):
    _tf_dataset_for_stmt(
        iter_, extra_test, body, get_state, set_state, symbol_names, opts)

  elif isinstance(iter_, iterator_ops.OwnedIterator):
    _tf_iterator_for_stmt(
        iter_, extra_test, body, get_state, set_state, symbol_names, opts)

  elif isinstance(iter_, ragged_tensor.RaggedTensor):
    _tf_ragged_for_stmt(
        iter_, extra_test, body, get_state, set_state, symbol_names, opts)

  elif isinstance(iter_, input_lib.DistributedIterator):
    raise NotImplementedError(
        'distributed iterators not supported yet, use the distributed dataset'
        ' directly')

  # TODO(mdan): Resolve the private access issue.
  elif isinstance(iter_, input_lib._IterableInput):  # pylint:disable=protected-access
    _tf_distributed_iterable_for_stmt(
        iter_, extra_test, body, get_state, set_state, symbol_names, opts)

  else:
    _py_for_stmt(iter_, extra_test, body, None, None)


def _py_for_stmt(iter_, extra_test, body, get_state, set_state):
  """Overload of for_stmt that executes a Python for loop."""
  del get_state, set_state

  if __debug__:
    checker = _PythonLoopChecker()
    before_iteration = checker.before_iteration
    after_iteration = checker.after_iteration
    before_iteration()

    original_body = body
    def protected_body(protected_iter):
      original_body(protected_iter)
      after_iteration()
      before_iteration()
    body = protected_body

  if extra_test is not None:
    if extra_test():
      for target in iter_:
        body(target)
        if not extra_test():
          break

  else:
    for target in iter_:
      body(target)


def _known_len_tf_for_stmt(
    iter_, extra_test, body, get_state, set_state, symbol_names, opts):
  """Overload of for_stmt that iterates over TF entities that admit a length."""
  n = py_builtins.len_(iter_)

  # TODO(b/117628877): Revisit performance once XLA has the necessary support.
  # Note: using a TensorArray creates an extra copy, but can calculate
  # gradients more efficiently than StridedSlice.
  ta = tensor_array_ops.TensorArray(iter_.dtype, size=n)
  iter_ = ta.unstack(iter_)

  iterate_index = compat_util.BasicRef(0)

  def aug_get_state():
    return (iterate_index.value,) + get_state()

  def aug_set_state(aug_loop_vars):
    # TOOD(mdan): Use starred assignment once we can switch to Py3-only syntax.
    iterate_index.value, loop_vars = aug_loop_vars[0], aug_loop_vars[1:]
    # The iteration index is not "output" by the for loop. If the iterate
    # is used outside the loop, it will appear in the loop vars separately.
    set_state(loop_vars)

  def aug_body():
    body(iter_.read(iterate_index.value))
    iterate_index.value += 1

  def aug_test():
    main_test = iterate_index.value < n
    if extra_test is not None:
      return control_flow_ops.cond(main_test, extra_test, lambda: False)
    return main_test

  opts['maximum_iterations'] = n

  _tf_while_stmt(
      aug_test,
      aug_body,
      aug_get_state,
      aug_set_state,
      ('<internal iterate>',) + symbol_names,
      opts,
  )


def _tf_ragged_for_stmt(
    iter_, extra_test, body, get_state, set_state, symbol_names, opts):
  """Overload of for_stmt that iterates over TF ragged tensors."""
  init_vars = get_state()
  _verify_loop_init_vars(init_vars, symbol_names)

  # TODO(mdan): Move this into len()? Requires eager support.
  if iter_.shape and iter_.shape[0] is not None:
    n = iter_.shape[0]
  else:
    n = iter_.row_lengths()[0]

  iterate_index = compat_util.BasicRef(0)

  def aug_get_state():
    return (iterate_index.value,) + get_state()

  def aug_set_state(aug_loop_vars):
    # TOOD(mdan): Use starred assignment once we can switch to Py3-only syntax.
    iterate_index.value, loop_vars = aug_loop_vars[0], aug_loop_vars[1:]
    # The iteration index is not "output" by the for loop. If the iterate
    # is used outside the loop, it will appear in the loop vars separately.
    set_state(loop_vars)

  def aug_body():
    body(iter_[iterate_index.value])
    iterate_index.value += 1

  def aug_test():
    main_test = iterate_index.value < n
    if extra_test is not None:
      return control_flow_ops.cond(main_test, extra_test, lambda: False)
    return main_test

  opts['maximum_iterations'] = n

  _tf_while_stmt(
      aug_test,
      aug_body,
      aug_get_state,
      aug_set_state,
      ('<internal iterate>',) + symbol_names,
      opts)


def _tf_range_for_stmt(
    iter_, extra_test, body, get_state, set_state, symbol_names, opts):
  """Overload of for_stmt that iterates over a TF range (and elides it)."""
  start, limit, delta = iter_.op.inputs

  iterate = compat_util.BasicRef(start)

  def aug_get_state():
    return (iterate.value,) + get_state()

  def aug_set_state(aug_loop_vars):
    # TOOD(mdan): Use starred assignment once we can switch to Py3-only syntax.
    iterate.value, loop_vars = aug_loop_vars[0], aug_loop_vars[1:]
    # The iteration index is not "output" by the for loop. If the iterate
    # is used outside the loop, it will appear in the loop vars separately.
    set_state(loop_vars)

  def aug_body():
    body(iterate.value)
    iterate.value += delta

  def aug_test():
    main_test = math_ops.logical_or(
        math_ops.logical_and(delta >= 0, iterate.value < limit),
        math_ops.logical_and(delta < 0, iterate.value > limit))
    if extra_test is not None:
      return control_flow_ops.cond(main_test, extra_test, lambda: False)
    return main_test

  opts['maximum_iterations'] = math_ops.cast(
      misc.get_range_len(start, limit, delta), dtypes.int32)

  _tf_while_stmt(
      aug_test,
      aug_body,
      aug_get_state,
      aug_set_state,
      ('<internal iterate>',) + symbol_names,
      opts)


def _tf_iterator_for_stmt(
    iter_, extra_test, body, get_state, set_state, symbol_names, opts):
  """Overload of for_stmt that iterates over TF Iterators. See for_loop."""
  symbol_names = ('<internal has_next>',) + symbol_names
  has_next = compat_util.BasicRef(True)

  def aug_get_state():
    return (has_next.value,) + get_state()

  def aug_set_state(aug_loop_vars):
    # TOOD(mdan): Use starred assignment once we can switch to Py3-only syntax.
    has_next.value, loop_vars = aug_loop_vars[0], aug_loop_vars[1:]
    set_state(loop_vars)

  init_vars = aug_get_state()
  _verify_loop_init_vars(init_vars, symbol_names)

  def aug_body():
    """Main body passed to _tf_while_stmt."""
    opt_iterate = iterator_ops.get_next_as_optional(iter_)
    has_next.value = opt_iterate.has_value()
    loop_vars = aug_get_state()  # updated by set_state() in _tf_while_loop.

    def main_path():
      body(opt_iterate.get_value())
      new_loop_vars = aug_get_state()
      # Note: this verification duplicates the one performed in tf_while_stmt,
      # but needs to be done earlier to prevent the tf.cond from blowing up
      # first.
      _verify_tf_loop_vars(
          init_vars, loop_vars, new_loop_vars, symbol_names, opts)
      return new_loop_vars

    def noop_path():
      return loop_vars

    # TODO(mdan): If tf.while_loop supported Optional, this could be avoided.
    # Calling set_state so that get_state() _tf_while_loop sees the conditional
    # tensors.
    aug_set_state(
        control_flow_ops.cond(has_next.value, main_path, noop_path))

  def aug_test():
    # This value takes a complicated path to get here:
    #   prev_iteration_body -> get_state -> tf.while_loop (as loop var)
    #   -> current_iteration_body -> set_state -> has_next.value
    main_test = has_next.value
    if extra_test is not None:
      return control_flow_ops.cond(main_test, extra_test, lambda: False)
    return main_test

  _tf_while_stmt(
      aug_test,
      aug_body,
      aug_get_state,
      aug_set_state,
      symbol_names,
      opts)


def _general_purpose_scan(ds, init_state, body):
  """Variant of Dataset.scan with semantics of general-purpose computation."""
  # Datasets are typically intended for data preprocessing. However, in
  # autograph loops they usually appear as general-purpose computations (for
  # example, a custom training loop). These two use cases require significantly
  # different optimization policies, the most important of which is the device
  # placement. The flag override for use_default_device below instructs the
  # runtime to treat the computation as general-purpose, rather than data
  # preprocessing.
  # TODO(mdan): s/use_default_device/specialize_for_input_pipeline.
  # TODO(mdan): Don't use private symbols.
  return scan_ops._ScanDataset(ds, init_state, body, use_default_device=False)  # pylint:disable=protected-access


def _tf_dataset_for_stmt(
    ds, extra_test, body, get_state, set_state, symbol_names, opts):
  """Overload of _dataset_for_stmt with early stopping. See for_stmt."""
  # Note: This is easier to follow with the insight that the computations in
  # a dataset pipeline are transposed (aka fused).
  # For example, given a pipeline input -> scan -> take_while -> reduce,
  # and a dataset with input [1, 2, 3], the computations occur in the following
  # order:
  #  reduce(take_while(scan(1)))
  #  reduce(take_while(scan(2)))
  #  reduce(take_while(scan(3)))

  init_vars = get_state()
  _verify_loop_init_vars(init_vars, symbol_names)

  # Workaround for Dataset.reduce not allowing empty state tensors - create
  # a dummy state variable that remains unused.
  # TODO(mdan): reduce should allow and match empty structures.
  if not init_vars:
    init_vars = (constant_op.constant(0),)
    symbol_names = ('<internal dummy>',)

    def dummy_set_state(unused_dummy):
      pass

    def dummy_get_state():
      return (constant_op.constant(0),)

    get_state, set_state = dummy_get_state, dummy_set_state

  def scan_body(scan_state, scan_inputs):
    """Main body of the Dataset.scan."""
    loop_vars, iterate = scan_state, scan_inputs
    set_state(loop_vars)

    def main_path():
      body(iterate)
      new_loop_vars = get_state()
      _verify_tf_loop_vars(
          init_vars, loop_vars, new_loop_vars, symbol_names, opts,
          check_shapes=False)
      return new_loop_vars

    if extra_test is not None:
      extra_cond = extra_test()
      new_loop_vars = control_flow_ops.cond(
          extra_cond, main_path, lambda: loop_vars)
    else:
      # TODO(mdan): the optimizer should be able to remove an invariant cond?
      extra_cond = (constant_op.constant(True),)  # dummy value, unused
      new_loop_vars = main_path()

    scan_outputs = new_loop_vars, extra_cond
    new_scan_state = new_loop_vars
    return new_scan_state, scan_outputs

  def take_while_predicate(unused_loop_vars, extra_cond):
    return extra_cond

  def reduce_body(unused_reduce_state, scan_outputs):
    output_loop_vars, unused_extra_cond = scan_outputs
    new_reduce_state = output_loop_vars
    return new_reduce_state

  ds = _general_purpose_scan(ds, init_vars, scan_body)
  if extra_test is not None:
    ds = ds.apply(take_while_ops.take_while(take_while_predicate))
  final_loop_vars = ds.reduce(init_vars, reduce_body)
  set_state(final_loop_vars)


def _tf_distributed_iterable_for_stmt(
    iter_, extra_test, body, get_state, set_state, symbol_names, opts):
  """Overload of for_stmt that iterates over TF distributed datasets."""

  if extra_test is not None:
    raise NotImplementedError(
        'break and return statements are not yet supported in '
        'for ... in distributed input loops.')

  init_vars = get_state()
  _verify_loop_init_vars(init_vars, symbol_names)

  if 'shape_invariants' in opts:
    opts['shape_invariants'] = _shape_invariants_mapping_to_positional_list(
        opts['shape_invariants'], init_vars)

  def reduce_body(loop_vars, iterate):
    set_state(loop_vars)
    body(iterate)
    new_loop_vars = get_state()
    _verify_tf_loop_vars(
        init_vars, loop_vars, new_loop_vars, symbol_names, opts)
    return new_loop_vars

  set_state(iter_.reduce(init_vars, reduce_body))


def while_stmt(test, body, get_state, set_state, symbol_names, opts):
  """Functional form of a while statement.

  The loop operates on a so-called state, which includes all symbols that are
  variant across loop iterations. In what follows we refer to state as either
  a tuple of entities that represent an actual state, or a list of arguments
  of the corresponding types.

  Args:
    test: Callable with the state as arguments, and boolean return type. The
      loop condition.
    body: Callable with the state as arguments, and state as return type. The
      actual loop body.
    get_state: Additional callable which can capture additional state (such as
      the values of composite symbols). This is only useful when staging the
      loop.
    set_state: Additional callable which save values captured by get_state back
      into the Python environment. This is only useful when staging the loop.
    symbol_names: Tuple containing the names of all loop variables.
    opts: Optional dict of extra loop parameters.

  Returns:
    Tuple containing the final state.
  """

  # Evaluate the initial test once in order to do the dispatch. The evaluation
  # is isolated to minimize unwanted side effects.
  # TODO(mdan): Do a full iteration - some state types might lower to Tensor.
  with func_graph.FuncGraph('tmp').as_default():
    init_test = test()

  # TensorFlow: Multiple evaluations are acceptable in this case, so we're fine
  # with the re-evaluation of `test` that `_tf_while_stmt` will make.
  if tensors.is_dense_tensor(init_test):
    _tf_while_stmt(test, body, get_state, set_state, symbol_names, opts)
    return

  # Normal Python: We already consumed one evaluation of `test`; consistently,
  # unroll one iteration before dispatching to a normal loop.
  # TODO(mdan): Push the "init_test" value via opts into _py_while_stmt?
  if not init_test:
    return
  body()

  _py_while_stmt(test, body, get_state, set_state, opts)


class _PythonLoopChecker(object):
  """Verifies Python loops for TF-specific limits."""

  __slots__ = (
      'iterations',
      'check_inefficient_unroll',
      'check_op_count_after_iteration',
      'ops_before_iteration',
      )

  def __init__(self):
    self.iterations = 1
    self.check_inefficient_unroll = WARN_INEFFICIENT_UNROLL

    # Triggered when we decided to test the op counts.
    self.check_op_count_after_iteration = False

  def _get_ops(self):
    return ops.get_default_graph().get_operations()

  def _check_unroll_limits(self):
    if self.iterations > PYTHON_MAX_ITERATIONS:
      raise ValueError('iteration limit exceeded')

  def _stop_checking_inefficient_unroll(self):
    self.check_inefficient_unroll = False
    self.check_op_count_after_iteration = False
    self.ops_before_iteration = None

  def _verify_inefficient_unroll(self):
    """Checks for possibly-inefficient creation of ops in a Python loop."""
    assert self.ops_before_iteration is not None
    ops_after_iteration = self._get_ops()
    new_ops = tuple(
        op for op in ops_after_iteration if op not in self.ops_before_iteration)

    if len(new_ops) < INEFFICIENT_UNROLL_MIN_OPS:
      return False

    ag_logging.warn(
        'Large unrolled loop detected. Did you mean to use a TF loop?'
        ' The following ops were created after iteration %s: %s'
        '\nSee'
        ' https://github.com/tensorflow/tensorflow/blob/master/'
        'tensorflow/python/autograph/g3doc/reference/common_errors.md'
        '#warning-large-unrolled-loop-detected'
        '\n'
        'Location:'
        '\n%s'
        '', self.iterations, new_ops, '\n'.join(traceback.format_stack()))
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

    if self.check_op_count_after_iteration:
      did_warn = self._verify_inefficient_unroll()
      if did_warn:
        self._stop_checking_inefficient_unroll()  # Only warn once.
      elif self.iterations > INEFFICIENT_UNROLL_MIN_ITERATIONS + 3:
        # Once deciding to check the op counts, only do it for a few iterations.
        self._stop_checking_inefficient_unroll()


def _py_while_stmt(test, body, get_state, set_state, opts):
  """Overload of while_stmt that executes a Python while loop."""
  del opts, get_state, set_state

  if __debug__:
    checker = _PythonLoopChecker()
    before_iteration = checker.before_iteration
    after_iteration = checker.after_iteration
    before_iteration()

    original_body = body
    def protected_body():
      original_body()
      after_iteration()
      before_iteration()
    body = protected_body

  while test():
    body()


def _shape_invariants_mapping_to_positional_list(mapping, keys):
  # The keys are not expected to be hashable.
  mapping = {id(k): (k, v) for k, v in mapping}
  result = []
  for k in keys:
    map_key, map_val = mapping.get(id(k), (None, None))
    result.append(map_val if map_key is k else None)
  return tuple(result)


def _tf_while_stmt(test, body, get_state, set_state, symbol_names, opts):
  """Overload of while_stmt that stages a TF while_stmt."""
  init_vars = get_state()
  _verify_loop_init_vars(init_vars, symbol_names)

  def aug_test(*loop_vars):
    set_state(loop_vars)
    return test()

  def aug_body(*loop_vars):
    set_state(loop_vars)
    body()
    new_loop_vars = get_state()
    _verify_tf_loop_vars(
        init_vars, loop_vars, new_loop_vars, symbol_names, opts)
    return new_loop_vars

  # Non-v2 while_loop unpacks the results when there is only one return value.
  # This enforces consistency across versions.
  opts['return_same_structure'] = True

  if 'shape_invariants' in opts:
    opts['shape_invariants'] = _shape_invariants_mapping_to_positional_list(
        opts['shape_invariants'], init_vars)

  final_loop_vars = control_flow_ops.while_loop(
      aug_test, aug_body, init_vars, **opts)
  set_state(final_loop_vars)


def if_stmt(cond,
            body,
            orelse,
            get_state,
            set_state,
            basic_symbol_names,
            composite_symbol_names):
  """Functional form of an if statement.

  Args:
    cond: Boolean.
    body: Callable with no arguments, and outputs of the positive (if) branch as
      return type.
    orelse: Callable with no arguments, and outputs of the negative (else)
      branch as return type.
    get_state: Function that returns a tuple containing the values of all
      composite symbols modified within the conditional. This allows access to
      state that branches may mutate through side effects. This function is not
      needed and should not be called when dispatching to code matching Python's
      default semantics. This is useful for checkpointing to avoid unintended
      side-effects when staging requires evaluating all code-paths.
    set_state: Function to set the values of all composite symbols modified
      within the conditional. This is the complement to get_state, used to
      restore checkpointed values. The single argument a tuple containing values
      for each composite symbol that may be modified in a branch of the
      conditional. The is usually the result of a call to get_state.
    basic_symbol_names: Tuple containing basic loop var names.
    composite_symbol_names: Tuple containing composite loop var names.

  Returns:
    Tuple containing the statement outputs.
  """
  # Note: tf.cond doesn't support SparseTensor.
  if tensors.is_dense_tensor(cond):
    return tf_if_stmt(cond, body, orelse, get_state, set_state,
                      basic_symbol_names, composite_symbol_names)
  else:
    return _py_if_stmt(cond, body, orelse)


def tf_if_stmt(cond, body, orelse, get_state, set_state, basic_symbol_names,
               composite_symbol_names):
  """Overload of if_stmt that stages a TF cond."""
  body = _wrap_disallow_undefs_from_cond(body, branch_name='if')
  orelse = _wrap_disallow_undefs_from_cond(orelse, branch_name='else')
  body = _isolate_state(body, get_state, set_state)
  orelse = _isolate_state(orelse, get_state, set_state)

  # `state` currently includes the values of any composite symbols (e.g. `a.b`)
  # composites modified by the loop. `final_vars` includes the values of basic
  # symbols (e.g. `a`) which cannot be passed by reference and must be returned.
  # See _isolate_state.
  # TODO(mdan): We should minimize calls to get/set_state.

  body_branch = 0
  orelse_branch = 1
  result = [None, None]

  def error_checking_body():
    result[body_branch] = body()
    if result[orelse_branch] is not None:
      _verify_tf_cond_vars(result[body_branch], result[orelse_branch],
                           basic_symbol_names + composite_symbol_names)
    return result[body_branch]

  def error_checking_orelse():
    result[orelse_branch] = orelse()
    if result[body_branch] is not None:
      _verify_tf_cond_vars(result[body_branch], result[orelse_branch],
                           basic_symbol_names + composite_symbol_names)
    return result[orelse_branch]

  final_vars, final_state = control_flow_ops.cond(cond, error_checking_body,
                                                  error_checking_orelse)

  set_state(final_state)

  return final_vars


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
    new_vars = func()
    # TODO(mdan): These should be copies, lest set_state might affect them.
    new_state = get_state()
    set_state(init_state)
    return new_vars, new_state

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


compat_util.deprecated_py2_support(__name__)
