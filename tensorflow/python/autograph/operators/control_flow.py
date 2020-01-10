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
from tensorflow.python.ops import array_ops
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

LIMIT_PYTHON_ITERATIONS = True
PYTHON_MAX_ITERATIONS = 100000000  # Fails in about one minute for empty loops.
WARN_INEFFICIENT_UNROLL = True
INEFFICIENT_UNROLL_MIN_ITERATIONS = 3000
INEFFICIENT_UNROLL_MIN_OPS = 1


# TODO(mdan): Use the custom operator pattern instead of type dispatch.
# An example of this pattern is found in the implementation of distributed
# datasets. Before it can be used though, we need to standardize the interface.


def _disallow_undefs_into_loop(*values):
  """Ensures that all values in the state are defined when entering a loop."""
  undefined = tuple(filter(special_values.is_undefined, values))
  if undefined:
    raise ValueError(
        'TensorFlow requires that the following symbols must be defined'
        ' before the loop: {}'.format(tuple(s.symbol_name for s in undefined)))

  for value in values:
    if special_values.is_undefined_return(value):
      # Assumption: the loop will only capture the variable which tracks the
      # return value if the loop contained a return statement.
      # TODO(mdan): This should be checked at the place where return occurs.
      raise ValueError(
          'return statements are not supported within a TensorFlow loop.')


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
  if isinstance(init, (bool, int, float, str, np.ndarray)):
    init = ops.convert_to_tensor_v2(init)
  if isinstance(entry, (bool, int, float, str, np.ndarray)):
    entry = ops.convert_to_tensor_v2(entry)
  if isinstance(exit_, (bool, int, float, str)):
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
            '"{}" has shape {} after the loop, which does not conform with'
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

  named_vars = zip(symbol_names, init_vars, iter_entry_vars, iter_exit_vars,
                   shape_invariants)
  for name, init, entry, exit_, invariant in named_vars:
    try:
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
  if isinstance(body_var, (bool, int, float, str)):
    body_var = ops.convert_to_tensor_v2(body_var)

  if isinstance(orelse_var, (bool, int, float, str)):
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

  # TODO(kkimlabs): Make this more consistent.
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


def for_stmt(iter_,
             extra_test,
             body,
             get_state,
             set_state,
             init_vars,
             basic_symbol_names,
             composite_symbol_names,
             opts):
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
    body: Callable with the iterate and the state as arguments, and state as
      return type. The actual loop body.
    get_state: Additional callable which can capture additional state (such as
      the values of composite symbols). This is only useful when staging the
      loop.
    set_state: Additional callable which save values captured by get_state back
      into the Python environment. This is only useful when staging the loop.
    init_vars: Tuple containing the initial state.
    basic_symbol_names: Tuple containing basic loop var names.
    composite_symbol_names: Tuple containing composite loop var names.
    opts: Optional dict of extra loop parameters.

  Returns:
    Tuple containing the final state.
  """
  if tensor_util.is_tensor(iter_):
    if tensors.is_range_tensor(iter_):
      return _tf_range_for_stmt(iter_, extra_test, body, get_state, set_state,
                                init_vars, basic_symbol_names,
                                composite_symbol_names, opts)
    else:
      return _known_len_tf_for_stmt(iter_, extra_test, body, get_state,
                                    set_state, init_vars, basic_symbol_names,
                                    composite_symbol_names, opts)

  if isinstance(iter_, dataset_ops.DatasetV2):
    return _tf_dataset_for_stmt(iter_, extra_test, body, get_state, set_state,
                                init_vars, basic_symbol_names,
                                composite_symbol_names, opts)

  if isinstance(iter_, iterator_ops.OwnedIterator):
    return _tf_iterator_for_stmt(iter_, extra_test, body, get_state, set_state,
                                 init_vars, basic_symbol_names,
                                 composite_symbol_names, opts)

  if isinstance(iter_, ragged_tensor.RaggedTensor):
    return _tf_ragged_for_stmt(iter_, extra_test, body, get_state, set_state,
                               init_vars, basic_symbol_names,
                               composite_symbol_names, opts)

  if isinstance(iter_, input_lib.DistributedIterator):
    raise NotImplementedError(
        'distributed iterators not supported yet, use the distributed dataset'
        ' directly')

  if isinstance(iter_, input_lib.DistributedDataset):
    return _tf_distributed_dataset_for_stmt(iter_, extra_test, body, init_vars)

  return _py_for_stmt(iter_, extra_test, body, get_state, set_state, init_vars)


def _py_for_stmt(iter_, extra_test, body, get_state, set_state, init_vars):
  """Overload of for_stmt that executes a Python for loop."""
  del get_state, set_state
  state = init_vars

  if extra_test is not None:
    if extra_test(*state):
      for target in iter_:
        state = body(target, *state)
        if not extra_test(*state):
          break

  else:
    for target in iter_:
      state = body(target, *state)

  return state


def _known_len_tf_for_stmt(iter_,
                           extra_test,
                           body,
                           get_state,
                           set_state,
                           init_vars,
                           basic_symbol_names,
                           composite_symbol_names,
                           opts):
  """Overload of for_stmt that iterates over TF entities that admit a length."""
  _disallow_undefs_into_loop(*init_vars)

  n = py_builtins.len_(iter_)
  # TODO(b/117628877): Revisit performance once XLA has the necessary support.
  # Note: using a TensorArray creates an extra copy, but can calculate
  # gradients more efficiently than StridedSlice.
  ta = tensor_array_ops.TensorArray(iter_.dtype, size=n)
  iter_ = ta.unstack(iter_)

  def while_body(iterate_index, *loop_vars):
    """Main loop body."""
    iterate = iter_.read(iterate_index)
    new_vars = body(iterate, *loop_vars)

    loop_vars = (iterate_index + 1,)
    if new_vars:
      loop_vars += new_vars

    return loop_vars

  def while_cond(iterate_index, *loop_vars):
    if extra_test is not None:
      return control_flow_ops.cond(iterate_index < n,
                                   lambda: extra_test(*loop_vars),
                                   lambda: False)
    return iterate_index < n

  opts['maximum_iterations'] = n

  results = _tf_while_stmt(
      while_cond,
      while_body,
      get_state,
      set_state,
      (array_ops.zeros_like(n),) + init_vars,
      ('<internal iterate>',) + basic_symbol_names,
      composite_symbol_names,
      opts,
  )

  # Note: the iteration index is not returned by the while loop, however
  # if a symbol with the same name exists outside the loop, it will be captured
  # by the loop variables and ultimately updated correctly.
  if isinstance(results, (tuple, list)):
    assert len(results) >= 1  # Has at least the iterate.
    if len(results) > 1:
      results = results[1:]
  else:
    results = ()

  return results


def _tf_ragged_for_stmt(iter_,
                        extra_test,
                        body,
                        get_state,
                        set_state,
                        init_vars,
                        basic_symbol_names,
                        composite_symbol_names,
                        opts):
  """Overload of for_stmt that iterates over TF ragged tensors."""
  _disallow_undefs_into_loop(*init_vars)

  # TODO(mdan): Move this into len()? Requires eager support.
  if iter_.shape and iter_.shape[0] is not None:
    n = iter_.shape[0]
  else:
    n = iter_.row_lengths()[0]

  opts['maximum_iterations'] = n

  def while_body(iterate_index, *loop_vars):
    """Main loop body."""
    iterate = iter_[iterate_index]
    new_vars = body(iterate, *loop_vars)

    loop_vars = (iterate_index + 1,)
    if new_vars:
      loop_vars += new_vars

    return loop_vars

  def while_cond(iterate_index, *loop_vars):
    if extra_test is not None:
      return control_flow_ops.cond(
          iterate_index < n,
          lambda: extra_test(*loop_vars),
          lambda: False,
      )
    return iterate_index < n

  opts['maximum_iterations'] = n

  results = _tf_while_stmt(
      while_cond,
      while_body,
      get_state,
      set_state,
      (array_ops.zeros_like(n),) + init_vars,
      ('<internal iterate>',) + basic_symbol_names,
      composite_symbol_names,
      opts,
  )

  if isinstance(results, (tuple, list)):
    assert len(results) >= 1  # Has at least the iterate.
    if len(results) > 1:
      results = results[1:]
  else:
    results = ()

  return results


def _tf_range_for_stmt(iter_,
                       extra_test,
                       body,
                       get_state,
                       set_state,
                       init_vars,
                       basic_symbol_names,
                       composite_symbol_names,
                       opts):
  """Overload of for_stmt that iterates over a TF range (and elides it)."""
  _disallow_undefs_into_loop(*init_vars)

  start, limit, delta = iter_.op.inputs

  def while_body(iterate, *loop_vars):
    new_vars = body(iterate, *loop_vars)
    loop_vars = (iterate + delta,)

    if new_vars:
      loop_vars += new_vars

    return loop_vars

  def while_cond(iterate, *loop_vars):
    """Cond function for `tf.while_loop`."""
    main_test = math_ops.logical_or(
        math_ops.logical_and(delta >= 0, iterate < limit),
        math_ops.logical_and(delta < 0, iterate > limit))
    if extra_test is not None:
      return control_flow_ops.cond(
          main_test,
          lambda: extra_test(*loop_vars),
          lambda: False,
      )
    return main_test

  opts['maximum_iterations'] = math_ops.cast(
      misc.get_range_len(start, limit, delta), dtypes.int32)

  results = _tf_while_stmt(
      while_cond,
      while_body,
      get_state,
      set_state,
      (start,) + init_vars,
      ('<internal iterate>',) + basic_symbol_names,
      composite_symbol_names,
      opts,
  )

  # Note: the iteration index is not returned by the while loop, however
  # if a symbol with the same name exists outside the loop, it will be captured
  # by the loop variables and ultimately updated correctly.
  if isinstance(results, (tuple, list)):
    assert len(results) >= 1  # Has at least the iterate.
    if len(results) > 1:
      results = results[1:]
  else:
    results = ()

  return results


def _tf_iterator_for_stmt(itr, extra_test, body, get_state, set_state,
                          init_vars, basic_symbol_names,
                          composite_symbol_names, opts):
  """Overload of for_stmt that iterates over TF Iterators. See for_loop."""
  _disallow_undefs_into_loop(*init_vars)

  def while_body_actual(opt_iterate, *loop_vars):
    """Actual main loop body."""
    new_vars = body(opt_iterate.get_value(), *loop_vars)
    # TODO(mdan): Fix this inconsistency in the converter.
    if new_vars is None:
      new_vars = ()
    # Note: this verification duplicates that perfrmed in tf_while_stmt,
    # but needs to be done earlier to prevent the tf.cond inside while_body
    # from blowing up first.
    _verify_tf_loop_vars(init_vars, loop_vars, new_vars,
                         basic_symbol_names + composite_symbol_names, opts)
    return new_vars

  def while_body(has_next, *loop_vars):
    """Main loop body."""
    opt_iterate = iterator_ops.get_next_as_optional(itr)
    has_next = opt_iterate.has_value()

    if not init_vars:
      # cond_v2 requires at least one state tensor in V1.
      dummy_state = (constant_op.constant(()),)
    else:
      dummy_state = ()

    # TODO(mdan): If tf.while_loop supported Optional, this could be avoided.
    new_vars = control_flow_ops.cond(
        has_next,
        lambda: dummy_state + while_body_actual(opt_iterate, *loop_vars),
        lambda: dummy_state + loop_vars,
    )

    if dummy_state:
      new_vars = new_vars[1:]

    return (has_next,) + new_vars

  def while_cond(has_next, *loop_vars):
    if extra_test is not None:
      return control_flow_ops.cond(
          has_next,
          lambda: extra_test(*loop_vars),
          lambda: False,
      )
    return has_next

  final_vars = _tf_while_stmt(
      while_cond,
      while_body,
      get_state,
      set_state,
      (True,) + init_vars,
      ('<internal has_next>',) + basic_symbol_names,
      composite_symbol_names,
      opts,
  )
  return final_vars[1:]


def _tf_dataset_for_stmt(ds, extra_test, body, get_state, set_state, init_vars,
                         basic_symbol_names, composite_symbol_names, opts):
  """Overload of for_stmt that iterates over TF Datasets."""
  _disallow_undefs_into_loop(*init_vars)

  if extra_test is not None:
    assert init_vars, 'Lowering should always add state.'
    return _dataset_for_stmt_with_extra_test(ds, extra_test, body, get_state,
                                             set_state, init_vars,
                                             basic_symbol_names,
                                             composite_symbol_names, opts)

  return _dataset_for_stmt_no_extra_test(ds, body, get_state, set_state,
                                         init_vars, basic_symbol_names,
                                         composite_symbol_names, opts)


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


def _dataset_for_stmt_with_extra_test(ds, extra_test, body, get_state,
                                      set_state, init_vars, basic_symbol_names,
                                      composite_symbol_names, opts):
  """Overload of _dataset_for_stmt with early stopping. See for_stmt."""

  # TODO(mdan): Simplify this - following it is extremely difficult.

  init_state = get_state()
  aug_init_vars = init_vars, init_state

  def scan_body(aug_vars, iterate):
    """The main loop body wrapper. Only calculates the stop condition."""
    loop_vars, state = aug_vars

    def true_fn():
      """Main path - stop condition is not set."""
      set_state(state)
      new_vars = body(iterate, *loop_vars)
      new_state = get_state()
      _verify_tf_loop_vars(
          init_vars + init_state,
          loop_vars + state,
          new_vars + new_state,
          basic_symbol_names + composite_symbol_names,
          opts,
          check_shapes=False)
      return new_vars, new_state

    extra_cond = extra_test(*loop_vars)
    new_vars, new_state = control_flow_ops.cond(
        extra_cond,
        true_fn,
        lambda: (loop_vars, state),
    )

    scan_outputs = new_vars, new_state, extra_cond
    # Note: new_aug_vars is the actual state of scan; scan_outputs is its output
    # (hence the redundancy).
    # get_state will pull any mutations that body may have made.
    new_aug_vars = new_vars, new_state
    return new_aug_vars, scan_outputs

  def take_while_predicate(unused_loop_vars, unused_state, extra_cond):
    return extra_cond

  def reduce_body(unused_aug_vars, scan_outputs):
    output_aug_vars, output_state, extra_cond = scan_outputs
    del extra_cond
    return output_aug_vars, output_state

  ds = _general_purpose_scan(ds, aug_init_vars, scan_body)
  ds = ds.apply(take_while_ops.take_while(take_while_predicate))
  final_aug_vars = ds.reduce(aug_init_vars, reduce_body)
  final_vars, final_state = final_aug_vars
  set_state(final_state)
  return final_vars


def _dataset_for_stmt_no_extra_test(ds, body, get_state, set_state, init_vars,
                                    basic_symbol_names, composite_symbol_names,
                                    opts):
  """Overload of _dataset_for_stmt without early stopping. See for_stmt."""
  init_state = get_state()
  assert isinstance(init_vars, tuple)
  assert isinstance(init_state, tuple)

  symbol_names = basic_symbol_names + composite_symbol_names

  # Workaround for Dataset.reduce not allowing empty state tensors - create
  # a dummy state variable that remains unused.
  # TODO(mdan): reduce should allow and match empty structures.
  no_vars = not init_vars
  no_state = not init_state

  if no_vars:
    init_vars = (constant_op.constant(0),)
    symbol_names = ('<internal dummy>',) + symbol_names
  if no_state:
    init_state = (constant_op.constant(0),)
    symbol_names = symbol_names + ('<internal dummy>',)

  def scan_body(aug_vars, iterate):
    """The main loop body wrapper."""
    loop_vars, state = aug_vars
    if not no_state:
      set_state(state)

    if no_vars:
      body(iterate)
      new_vars = loop_vars
    else:
      new_vars = body(iterate, *loop_vars)

    if no_state:
      new_state = state
    else:
      new_state = get_state()

    _verify_tf_loop_vars(
        init_vars + init_state,
        loop_vars + state,
        new_vars + new_state,
        symbol_names,
        opts,
        check_shapes=False)

    scan_outputs = new_vars, new_state
    # Note: new_aug_vars is the actual state of scan; scan_outputs is its output
    # (hence the redundancy).
    # get_state will pull any mutations that body may have made.
    new_aug_vars = new_vars, new_state
    return new_aug_vars, scan_outputs

  def reduce_body(unused_aug_vars, scan_outputs):
    output_aug_vars, output_state = scan_outputs
    return output_aug_vars, output_state

  aug_vars = init_vars, get_state()
  ds = _general_purpose_scan(ds, aug_vars, scan_body)
  final_vars, final_state = ds.reduce(aug_vars, reduce_body)
  set_state(final_state)

  if no_vars:
    return ()
  return final_vars


def _tf_distributed_dataset_for_stmt(iter_, extra_test, body, init_state):
  """Overload of for..in statement that iterates over the input."""
  _disallow_undefs_into_loop(*init_state)

  if extra_test is not None:
    raise NotImplementedError(
        'break and return statements are not yet supported in '
        'for ... in distributed input loops.')

  def reduce_body(state, iterate):
    new_state = body(iterate, *state)
    return new_state

  if init_state:
    return iter_.reduce(init_state, reduce_body)

  # TODO(anjalisridhar): This is a workaround for Dataset.reduce not allowing
  # empty state tensors - create a dummy state variable that remains unused.
  # Identify if we need this workaround and remove if unnecessary.
  def reduce_body_with_dummy_state(state, iterate):
    reduce_body((), iterate)
    return state
  iter_.reduce((constant_op.constant(0),), reduce_body_with_dummy_state)
  return ()


def while_stmt(test,
               body,
               get_state,
               set_state,
               init_vars,
               basic_symbol_names,
               composite_symbol_names,
               opts):
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
    init_vars: Tuple containing the initial state.
    basic_symbol_names: Tuple containing basic loop var names.
    composite_symbol_names: Tuple containing composite loop var names.
    opts: Optional dict of extra loop parameters.

  Returns:
    Tuple containing the final state.
  """

  # Evaluate the initial test once in order to do the dispatch. The evaluation
  # is isolated to minimize unwanted side effects.
  # TODO(mdan): Do a full iteration - some state types might lower to Tensor.
  with func_graph.FuncGraph('tmp').as_default():
    init_test = test(*init_vars)

  # TensorFlow: Multiple evaluations are acceptable in this case, so we're fine
  # with the re-evaluation of `test` that `_tf_while_stmt` will make.
  if tensors.is_dense_tensor(init_test):
    return _tf_while_stmt(test, body, get_state, set_state, init_vars,
                          basic_symbol_names, composite_symbol_names, opts)

  # Normal Python: We already consumed one evaluation of `test`; consistently,
  # unroll one iteration before dispatching to a normal loop.
  # TODO(mdan): Push the "init_test" value via opts into _py_while_stmt?
  if not init_test:
    return init_vars
  init_vars = body(*init_vars)

  return _py_while_stmt(test, body, get_state, set_state, init_vars, opts)


def _shape_invariants_mapping_to_positional_list(mapping, keys):
  # The keys are not expected to be hashable.
  mapping = {id(k): (k, v) for k, v in mapping}
  result = []
  for k in keys:
    map_key, map_val = mapping.get(id(k), (None, None))
    result.append(map_val if map_key is k else None)
  return tuple(result)


def _tf_while_stmt(test, body, get_state, set_state, init_vars,
                   basic_symbol_names, composite_symbol_names, opts):
  """Overload of while_stmt that stages a TF while_stmt."""
  _disallow_undefs_into_loop(*init_vars)

  aug_init_vars = init_vars + get_state()

  # TODO(mdan): Simplify this.
  loop_vars_slice = slice(len(init_vars))
  state_slice = slice(len(init_vars), None)

  def aug_test(*aug_loop_vars):
    state = aug_loop_vars[state_slice]
    set_state(state)
    return test(*aug_loop_vars[loop_vars_slice])

  def aug_body(*aug_loop_vars):
    """Main loop body."""
    state = aug_loop_vars[state_slice]
    set_state(state)
    loop_vars = body(*aug_loop_vars[loop_vars_slice])
    new_state = loop_vars + get_state()
    _verify_tf_loop_vars(aug_init_vars, aug_loop_vars, new_state,
                         basic_symbol_names + composite_symbol_names, opts)

    return new_state

  # Non-v2 while_loop unpacks the results when there is only one return value.
  # This enforces consistency across versions.
  opts['return_same_structure'] = True

  if 'shape_invariants' in opts:
    opts['shape_invariants'] = _shape_invariants_mapping_to_positional_list(
        opts['shape_invariants'], aug_init_vars)

  final_aug_vars = control_flow_ops.while_loop(aug_test, aug_body,
                                               aug_init_vars, **opts)
  final_state = final_aug_vars[state_slice]
  set_state(final_state)
  return final_aug_vars[loop_vars_slice]


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


def _py_while_stmt(test, body, get_state, set_state, init_vars, opts):
  """Overload of while_stmt that executes a Python while loop."""
  del opts, get_state, set_state

  if __debug__:
    checker = _PythonLoopChecker()

  loop_vars = init_vars
  while test(*loop_vars):

    if __debug__:
      checker.before_iteration()

    loop_vars = body(*loop_vars)

    if __debug__:
      checker.after_iteration()

  return loop_vars


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
