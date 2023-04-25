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
"""Cond function for Control Flow Operations."""

from tensorflow.python.eager import context
from tensorflow.python.eager.polymorphic_function import eager_function_run
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util as util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import core
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export

# TODO(b/269483538): below lazy loads
#   needed for references while refactors are in progress
control_flow_ops = LazyLoader(
    "control_flow_ops", globals(),
    "tensorflow.python.ops.control_flow_ops")
# This is to avoid a circular dependency:
# cond_v2 -> gradients_util -> control_flow_ops
cond_v2 = LazyLoader("cond_v2", globals(),
                     "tensorflow.python.ops.cond_v2")


# pylint: disable=redefined-outer-name
# pylint: disable=g-doc-args
@tf_export(v1=["cond"])
@dispatch.add_dispatch_support
@deprecation.deprecated_args(
    None, "fn1/fn2 are deprecated in favor of the true_fn/false_fn arguments.",
    "fn1", "fn2")
def cond(pred,
         true_fn=None,
         false_fn=None,
         strict=False,
         name=None,
         fn1=None,
         fn2=None):
  """Return `true_fn()` if the predicate `pred` is true else `false_fn()`.

  `true_fn` and `false_fn` both return lists of output tensors. `true_fn` and
  `false_fn` must have the same non-zero number and type of outputs.

  **WARNING**: Any Tensors or Operations created outside of `true_fn` and
  `false_fn` will be executed regardless of which branch is selected at runtime.

  Although this behavior is consistent with the dataflow model of TensorFlow,
  it has frequently surprised users who expected a lazier semantics.
  Consider the following simple program:

  ```python
  z = tf.multiply(a, b)
  result = tf.cond(x < y, lambda: tf.add(x, z), lambda: tf.square(y))
  ```

  If `x < y`, the `tf.add` operation will be executed and `tf.square`
  operation will not be executed. Since `z` is needed for at least one
  branch of the `cond`, the `tf.multiply` operation is always executed,
  unconditionally.

  Note that `cond` calls `true_fn` and `false_fn` *exactly once* (inside the
  call to `cond`, and not at all during `Session.run()`). `cond`
  stitches together the graph fragments created during the `true_fn` and
  `false_fn` calls with some additional graph nodes to ensure that the right
  branch gets executed depending on the value of `pred`.

  `tf.cond` supports nested structures as implemented in
  `tensorflow.python.util.nest`. Both `true_fn` and `false_fn` must return the
  same (possibly nested) value structure of lists, tuples, and/or named tuples.
  Singleton lists and tuples form the only exceptions to this: when returned by
  `true_fn` and/or `false_fn`, they are implicitly unpacked to single values.
  This behavior is disabled by passing `strict=True`.

  Args:
    pred: A scalar determining whether to return the result of `true_fn` or
      `false_fn`.
    true_fn: The callable to be performed if pred is true.
    false_fn: The callable to be performed if pred is false.
    strict: A boolean that enables/disables 'strict' mode; see above.
    name: Optional name prefix for the returned tensors.

  Returns:
    Tensors returned by the call to either `true_fn` or `false_fn`. If the
    callables return a singleton list, the element is extracted from the list.

  Raises:
    TypeError: if `true_fn` or `false_fn` is not callable.
    ValueError: if `true_fn` and `false_fn` do not return the same number of
      tensors, or return tensors of different types.

  Example:

  ```python
  x = tf.constant(2)
  y = tf.constant(5)
  def f1(): return tf.multiply(x, 17)
  def f2(): return tf.add(y, 23)
  r = tf.cond(tf.less(x, y), f1, f2)
  # r is set to f1().
  # Operations in f2 (e.g., tf.add) are not executed.
  ```

  """
  # We needed to make true_fn/false_fn keyword arguments for
  # backwards-compatibility. This check exists so that we can convert back to
  # having them be positional arguments.
  # TODO(joshl): Make `true_fn` and `false_fn` positional arguments after
  # `fn1` and `fn2` are deleted.
  if fn1 is not None:
    if true_fn is not None:
      raise TypeError(
          "cond(): 'true_fn' and 'fn1' may not be set simultaneously.")
    true_fn = fn1
  elif true_fn is None:
    raise TypeError("cond(): 'true_fn' argument required")
  if fn2 is not None:
    if false_fn is not None:
      raise TypeError(
          "cond(): 'false_fn' and 'fn2' may not be set simultaneously.")
    false_fn = fn2
  elif false_fn is None:
    raise TypeError("cond(): 'false_fn' argument required")

  if not callable(true_fn):
    raise TypeError("'true_fn' must be callable.")
  if not callable(false_fn):
    raise TypeError("'false_fn' must be callable.")

  if context.executing_eagerly():
    return _eager_cond_implementation(pred, true_fn, false_fn, strict, name)

  # Always enable control flow v2 if building a function, regardless of toggle.
  if util.EnableControlFlowV2(ops.get_default_graph()):
    return cond_v2.cond_v2(pred, true_fn, false_fn, name)

  with ops.name_scope(name, "cond", [pred]):
    # Add the Switch to the graph.
    if isinstance(pred, bool):
      raise TypeError("'pred' must not be a Python bool.")
    p_2, p_1 = control_flow_ops.switch(pred, pred)
    pivot_1 = array_ops.identity(p_1, name="switch_t")
    pivot_2 = array_ops.identity(p_2, name="switch_f")
    pred = array_ops.identity(pred, name="pred_id")
    # Disable the fetching of tensors that are only on one branch of cond.
    for tensor in [p_1, p_2, pivot_1, pivot_2, pred]:
      tensor.op.graph.prevent_fetching(tensor.op)

    # Build the graph for the true branch in a new context.
    context_t = control_flow_ops.CondContext(pred, pivot_1, branch=1)
    try:
      context_t.Enter()
      orig_res_t, res_t = context_t.BuildCondBranch(true_fn)
      if orig_res_t is None:
        raise ValueError("'true_fn' must have a return value.")
      context_t.ExitResult(res_t)
    finally:
      context_t.Exit()

    # Build the graph for the false branch in a new context.
    context_f = control_flow_ops.CondContext(pred, pivot_2, branch=0)
    try:
      context_f.Enter()
      orig_res_f, res_f = context_f.BuildCondBranch(false_fn)
      if orig_res_f is None:
        raise ValueError("'false_fn' must have a return value.")
      context_f.ExitResult(res_f)
    finally:
      context_f.Exit()

    if not strict:
      orig_res_t = _UnpackIfSingleton(orig_res_t)
      orig_res_f = _UnpackIfSingleton(orig_res_f)

    # Check that the return values of the two branches have the same structure.
    try:
      nest.assert_same_structure(orig_res_t, orig_res_f, expand_composites=True)
    except (TypeError, ValueError):
      nest.map_structure(_cast_indexed_slice_indices, orig_res_t, orig_res_f)
      nest.map_structure(_cast_indexed_slice_indices, res_t, res_f)
      try:
        nest.assert_same_structure(orig_res_t, orig_res_f,
                                   expand_composites=True)
      except TypeError as e:
        raise TypeError(
            f"Incompatible return types of 'true_fn' and 'false_fn': {e}")
      except ValueError as e:
        raise ValueError(
            f"Incompatible return values of 'true_fn' and 'false_fn': {e}")

    # Add the final merge to the graph.
    if not res_t:
      raise ValueError(
          "'true_fn' and 'false_fn' must return at least one result.")

    res_t_flat = nest.flatten(res_t, expand_composites=True)
    res_f_flat = nest.flatten(res_f, expand_composites=True)

    for (x, y) in zip(res_t_flat, res_f_flat):
      assert isinstance(x, ops.Tensor) and isinstance(y, ops.Tensor)
      if x.dtype.base_dtype != y.dtype.base_dtype:
        raise ValueError(
            "Outputs of 'true_fn' and 'false_fn' must have the same type(s). "
            f"Received {x.dtype.name} from 'true_fn' "
            f"and {y.dtype.name} from 'false_fn'.")

    merges = [
        control_flow_ops.merge(pair)[0] for pair in zip(res_f_flat, res_t_flat)]
    merges = nest.map_structure(
        control_flow_ops._convert_flow_to_tensorarray,  # pylint: disable=protected-access
        nest.flatten(orig_res_t, expand_composites=True),
        merges)

    # Only add non-nested conds to the collection. Any nested control flow will
    # be encapsulated in the root context.
    assert context_t.outer_context == context_f.outer_context
    if context_t.outer_context is None:
      ops.add_to_collection(ops.GraphKeys.COND_CONTEXT, context_t)
      ops.add_to_collection(ops.GraphKeys.COND_CONTEXT, context_f)

    merges = nest.pack_sequence_as(
        structure=orig_res_t, flat_sequence=merges, expand_composites=True)

    # Singleton lists and tuples are automatically unpacked if strict == False.
    if not strict:
      merges = _UnpackIfSingleton(merges)
    return merges


@tf_export("cond", v1=[])
@dispatch.add_dispatch_support
def cond_for_tf_v2(pred, true_fn=None, false_fn=None, name=None):
  """Return `true_fn()` if the predicate `pred` is true else `false_fn()`.

  Note: This op is automatically used in a `tf.function` to convert Python
  if-statements when the predicate is a `tf.Tensor`, unless `autograph=False` is
  explicitly specified in `tf.function` args. For example, the following are
  equivalent:

  >>> @tf.function
  ... def fun1(x,y):
  ...   if x > 0:  # AutoGraph converts if-statement to tf.cond().
  ...     z = y+1
  ...   else:
  ...     z = y-1
  ...   return z
  >>> fun1(tf.constant(7), tf.constant(3)).numpy()
  4

  >>> @tf.function
  ... def fun2(x,y):
  ...   pred = x > 0
  ...   true_fn =  lambda: y+1
  ...   false_fn = lambda: y-1
  ...   return tf.cond(pred, true_fn, false_fn)  # Use tf.cond() explicitly.
  >>> fun1(tf.constant(7), tf.constant(3)).numpy()
  4

  For more information, see [tf.function and AutoGraph guide](
  https://www.tensorflow.org/guide/function#autograph_transformations).

  `true_fn` and `false_fn` both return lists of output tensors. `true_fn` and
  `false_fn` must have the same non-zero number and type of outputs.

  **WARNING**: Any Tensors or Operations created outside of `true_fn` and
  `false_fn` will be executed regardless of which branch is selected at runtime.

  Although this behavior is consistent with the dataflow model of TensorFlow,
  it has frequently surprised users who expected a lazier semantics.
  Consider the following simple program:

  >>> x, y = tf.constant(2, dtype=tf.int32), tf.constant(4, dtype=tf.int32)
  >>> z = tf.multiply(x, y)
  >>> r = tf.cond(x < y, lambda: tf.add(x, z), lambda: tf.square(y))
  >>> r.numpy()
  10

  If `x < y`, the `tf.add` operation will be executed and `tf.square`
  operation will not be executed. Since `z` is needed for at least one
  branch of the `cond`, the `tf.multiply` operation is always executed,
  unconditionally.

  Note that `cond` calls `true_fn` and `false_fn` *exactly once* (inside the
  call to `cond`, and not at all during `Session.run()`). `cond`
  stitches together the graph fragments created during the `true_fn` and
  `false_fn` calls with some additional graph nodes to ensure that the right
  branch gets executed depending on the value of `pred`.

  `tf.cond` supports nested structures as implemented in
  `tensorflow.python.util.nest`. Both `true_fn` and `false_fn` must return the
  same (possibly nested) value structure of lists, tuples, and/or named tuples.
  Singleton lists and tuples form the only exceptions to this: when returned by
  `true_fn` and/or `false_fn`, they are implicitly unpacked to single values.

  Note: It is illegal to "directly" use tensors created inside a cond branch
  outside it, e.g. by storing a reference to a branch tensor in the python
  state. If you need to use a tensor created in a branch function you should
  return it as an output of the branch function and use the output from
  `tf.cond` instead.

  Args:
    pred: A scalar determining whether to return the result of `true_fn` or
      `false_fn`.
    true_fn: The callable to be performed if pred is true.
    false_fn: The callable to be performed if pred is false.
    name: Optional name prefix for the returned tensors.

  Returns:
    Tensors returned by the call to either `true_fn` or `false_fn`. If the
    callables return a singleton list, the element is extracted from the list.

  Raises:
    TypeError: if `true_fn` or `false_fn` is not callable.
    ValueError: if `true_fn` and `false_fn` do not return the same number of
      tensors, or return tensors of different types.

  Example:

  >>> x = tf.constant(2)
  >>> y = tf.constant(5)
  >>> def f1(): return tf.multiply(x, 7)
  >>> def f2(): return tf.add(y, 3)
  >>> r = tf.cond(tf.less(x, y), f1, f2)
  >>> # r is set to f1().
  >>> # Operations in f2 (e.g., tf.add) are not executed.
  >>> r.numpy()
  14

  """
  return cond(pred, true_fn=true_fn, false_fn=false_fn, strict=True, name=name)


def _UnpackIfSingleton(res):
  if isinstance(res, (list, tuple)) and len(res) == 1:
    return res[0]
  else:
    return res


def _eager_cond_implementation(pred, true_fn, false_fn, strict, name):
  """Special cases for `cond` when executing eagerly."""
  pred = ops.convert_to_tensor(pred)
  pred_constant_value = tensor_util.constant_value(pred)
  if pred_constant_value is None:
    # Eager tensors from a parallel device may not have a constant
    # value. Running the cond op itself would work, but we don't have logic to
    # build cond ops without wrapping in a function first.
    if (not isinstance(true_fn, core.GenericFunction)
        or not isinstance(false_fn, core.GenericFunction)):
      raise TypeError("When running tf.cond on a parallel device, 'true_fn' "
                      "and 'false_fn' must be decorated with `tf.function`.")
    functions_run_eagerly = eager_function_run.functions_run_eagerly()
    if functions_run_eagerly:
      # We need to use tf.function to deal with variable creation inside the
      # cond, and skipping it because of run_functions_eagerly would just
      # crash immediately.
      logging.warning(
          "It looks like tf.function behavior was disabled, perhaps using "
          "tf.config.run_functions_eagerly. Parallelized tf.cond requires "
          "tf.function to work. This primitive will override the disable.")
    eager_function_run.run_functions_eagerly(False)
    try:
      return cond_v2.cond_v2(pred, true_fn, false_fn, name)
    finally:
      if functions_run_eagerly is not None:
        eager_function_run.run_functions_eagerly(functions_run_eagerly)
  else:
    # For conditions which are eager tensors with a constant value (most of
    # them), we only call the relevant branch function and execute it eagerly.
    with ops.name_scope(name, "cond", [pred]):
      if pred_constant_value:
        result = true_fn()
      else:
        result = false_fn()
      if not strict:
        result = _UnpackIfSingleton(result)
      return result


def _cast_indexed_slice_indices(a, b):
  """Cast IndexedSlice.indices from int32 to int64 where necessary.

  If `a` and `b` are both IndexedSlices, and their indices have different
  dtypes, then cast both their dtypes to `int64` (modifies `a` and `b`
  in-place).  Otherwise, does nothing.

  Args:
    a: A value, which may be an IndexedSlices.
    b: A value, which may be an IndexedSlices.
  """
  if (isinstance(a, indexed_slices.IndexedSlices) and
      isinstance(b, indexed_slices.IndexedSlices) and
      a.indices.dtype != b.indices.dtype):
    # pylint: disable=protected-access
    a._indices = math_ops.cast(a.indices, dtypes.int64)
    b._indices = math_ops.cast(b.indices, dtypes.int64)
