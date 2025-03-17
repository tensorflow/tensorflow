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
"""While loop for Control Flow Operations."""

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util as util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_v2
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export


# @TODO(b/133606651) Replace "shape_invariants" with "loop_vars_signature".
# pylint: disable=redefined-outer-name
@tf_export("while_loop", v1=[])
@deprecation.deprecated_arg_values(
    None,
    """back_prop=False is deprecated. Consider using tf.stop_gradient instead.
Instead of:
results = tf.while_loop(c, b, vars, back_prop=False)
Use:
results = tf.nest.map_structure(tf.stop_gradient, tf.while_loop(c, b, vars))""",
    warn_once=True,
    back_prop=False)
def while_loop_v2(cond,
                  body,
                  loop_vars,
                  shape_invariants=None,
                  parallel_iterations=10,
                  back_prop=True,
                  swap_memory=False,
                  maximum_iterations=None,
                  name=None):
  """Repeat `body` while the condition `cond` is true.

  Note: This op is automatically used in a `tf.function` to convert Python for-
  and while- loops when the loop variable is a `tf.Tensor`, unless
  `autograph=False` is explicitly specified in `tf.function` args. For example,
  the following are equivalent:

  >>> @tf.function
  ... def sumSquare(n):
  ...   i, result = tf.constant(0), tf.constant(0)
  ...   while i < n: # AutoGraph converts while-loop to tf.while_loop().
  ...     result += i * i
  ...     i += 1
  ...   return result
  >>> print(sumSquare(10).numpy())
  285

  >>> @tf.function
  ... def sumSquare2(n):
  ...   i, result = tf.constant(0), tf.constant(0)
  ...   c = lambda i, _: tf.less(i, n)
  ...   b = lambda i, result: (i + 1, result + i * i)
  ...   return tf.while_loop(c, b, [i, result])[1]
  >>> print(sumSquare2(10).numpy())
  285

  For more information, see [tf.function and AutoGraph guide
  ](https://www.tensorflow.org/guide/function#autograph_transformations).

  `cond` is a callable returning a boolean scalar tensor. `body` is a callable
  returning a (possibly nested) tuple, namedtuple or list of tensors of the same
  arity (length and structure) and types as `loop_vars`. `loop_vars` is a
  (possibly nested) tuple, namedtuple or list of tensors that is passed to both
  `cond` and `body`. `cond` and `body` both take as many arguments as there are
  `loop_vars`.

  In addition to regular Tensors or IndexedSlices, the body may accept and
  return TensorArray objects.  The flows of the TensorArray objects will
  be appropriately forwarded between loops and during gradient calculations.

  Note that `while_loop` calls `cond` and `body` *exactly once* (inside the
  call to `while_loop`, and not at all during `Session.run()`). `while_loop`
  stitches together the graph fragments created during the `cond` and `body`
  calls with some additional graph nodes to create the graph flow that
  repeats `body` until `cond` returns false.

  For correctness, `tf.while_loop()` strictly enforces shape invariants for
  the loop variables. A shape invariant is a (possibly partial) shape that
  is unchanged across the iterations of the loop. An error will be raised
  if the shape of a loop variable after an iteration is determined to be more
  general than or incompatible with its shape invariant. For example, a shape
  of `[11, None]` is more general than a shape of `[11, 17]`, and `[11, 21]` is
  not compatible with `[11, 17]`. By default (if the argument `shape_invariants`
  is not specified), it is assumed that the initial shape of each tensor in
  `loop_vars` is the same in every iteration. The `shape_invariants` argument
  allows the caller to specify a less specific shape invariant for each loop
  variable, which is needed if the shape varies between iterations. The
  `tf.Tensor.set_shape`
  function may also be used in the `body` function to indicate that
  the output loop variable has a particular shape. The shape invariant for
  SparseTensor and IndexedSlices are treated specially as follows:

  a) If a loop variable is a SparseTensor, the shape invariant must be
  `TensorShape([r])` where `r` is the rank of the dense tensor represented
  by the sparse tensor. It means the shapes of the three tensors of the
  SparseTensor are `([None], [None, r], [r])`. NOTE: The shape invariant here
  is the shape of the SparseTensor.dense_shape property. It must be the shape of
  a vector.

  b) If a loop variable is an IndexedSlices, the shape invariant must be
  a shape invariant of the values tensor of the IndexedSlices. It means
  the shapes of the three tensors of the IndexedSlices are `(shape, [shape[0]],
  [shape.ndims])`.

  `while_loop` implements non-strict semantics, enabling multiple iterations
  to run in parallel. The maximum number of parallel iterations can be
  controlled by `parallel_iterations`, which gives users some control over
  memory consumption and execution order. For correct programs, `while_loop`
  should return the same result for any `parallel_iterations > 0`.

  For training, TensorFlow stores the tensors that are produced in the
  forward inference and are needed in back propagation. These tensors are a
  main source of memory consumption and often cause OOM errors when training
  on GPUs. When the flag swap_memory is true, we swap out these tensors from
  GPU to CPU. This for example allows us to train RNN models with very long
  sequences and large batches.

  Args:
    cond: A callable that represents the termination condition of the loop.
    body: A callable that represents the loop body.
    loop_vars: A (possibly nested) tuple, namedtuple or list of numpy array,
      `Tensor`, and `TensorArray` objects.
    shape_invariants: The shape invariants for the loop variables.
    parallel_iterations: The number of iterations allowed to run in parallel. It
      must be a positive integer.
    back_prop: (optional) Deprecated. False disables support for back
      propagation. Prefer using `tf.stop_gradient` instead.
    swap_memory: Whether GPU-CPU memory swap is enabled for this loop.
    maximum_iterations: Optional maximum number of iterations of the while loop
      to run.  If provided, the `cond` output is AND-ed with an additional
      condition ensuring the number of iterations executed is no greater than
      `maximum_iterations`.
    name: Optional name prefix for the returned tensors.

  Returns:
    The output tensors for the loop variables after the loop. The return value
      has the same structure as `loop_vars`.

  Raises:
    TypeError: if `cond` or `body` is not callable.
    ValueError: if `loop_vars` is empty.

  Example:

  >>> i = tf.constant(0)
  >>> c = lambda i: tf.less(i, 10)
  >>> b = lambda i: (tf.add(i, 1), )
  >>> r = tf.while_loop(c, b, [i])[0]
  >>> print(r.numpy())
  10

  Example with nesting and a namedtuple:

  >>> import collections
  >>> Pair = collections.namedtuple('Pair', 'j, k')
  >>> ijk_0 = (tf.constant(0), Pair(tf.constant(1), tf.constant(2)))
  >>> c = lambda i, p: i < 10
  >>> b = lambda i, p: (i + 1, Pair((p.j + p.k), (p.j - p.k)))
  >>> ijk_final = tf.while_loop(c, b, ijk_0)[1]
  >>> ijk_final[0].numpy().item(), ijk_final[1].numpy().item()
  (32, 64)

  Example using shape_invariants:

  >>> i0 = tf.constant(0)
  >>> m0 = tf.ones([2, 2])
  >>> c = lambda i, m: i < 10
  >>> b = lambda i, m: [i+1, tf.concat([m, m], axis=0)]
  >>> tf.while_loop(
  ...     c, b, loop_vars=[i0, m0],
  ...     shape_invariants=[i0.get_shape(), tf.TensorShape([None, 2])])[1]
  <tf.Tensor: shape=(2048, 2), dtype=float32, numpy=...>

  Example which demonstrates non-strict semantics: In the following
  example, the final value of `counter` does not depend on `x`. So
  the `while_loop` can increment the counter parallel to updates of `x`.
  However, because the loop counter at one loop iteration depends
  on the value at the previous iteration, the loop counter itself cannot
  be incremented in parallel. Hence if we just want the final value of the
  counter (which we print on the line `print(sess.run(i))`), then
  `x` will never be incremented, but the counter will be updated on a
  single thread. Conversely, if we want the value of the output (which we
  print on the line `print(sess.run(out).shape)`), then the counter may be
  incremented on its own thread, while `x` can be incremented in
  parallel on a separate thread. In the extreme case, it is conceivable
  that the thread incrementing the counter runs until completion before
  `x` is incremented even a single time. The only thing that can never
  happen is that the thread updating `x` can never get ahead of the
  counter thread because the thread incrementing `x` depends on the value
  of the counter.

  >>> with tf.compat.v1.Session() as sess:
  ...   n = 10
  ...   c = lambda i, x: i < n
  ...   b = lambda i, x: (
  ...       tf.compat.v1.Print(i + 1, [i], "Updating i based on i == "),
  ...       # Let x depend on i
  ...       tf.compat.v1.Print(x + i, [i], "Updating x based on i == "))
  ...
  ...   # Make x to be a big matrix so its updating thread would run slowly
  ...   x = tf.zeros([1000, 100], dtype=tf.int32)
  ...   counter = tf.constant(0)
  ...   counter_out, x_out = tf.while_loop(c, b, (counter, x))
  ...
  ...   # The following line may increment the counter and x in parallel.
  ...   # The counter thread may get ahead of the x thread, but not the
  ...   # other way around. For example, the log may contain these messages:
  ...   # ```
  ...   # Updating i based on i == [9]
  ...   # Updating x based on i == [3]
  ...   # ```
  ...   # meaning that the counter(i) thread is on iteration 9,
  ...   # while the x thread is on iteration 3.
  ...   print(sess.run(x_out).shape)
  (1000, 100)

  """
  return while_loop(
      cond=cond,
      body=body,
      loop_vars=loop_vars,
      shape_invariants=shape_invariants,
      parallel_iterations=parallel_iterations,
      back_prop=back_prop,
      swap_memory=swap_memory,
      name=name,
      maximum_iterations=maximum_iterations,
      return_same_structure=True)


# pylint: disable=redefined-outer-name
@tf_export(v1=["while_loop"])
def while_loop(cond,
               body,
               loop_vars,
               shape_invariants=None,
               parallel_iterations=10,
               back_prop=True,
               swap_memory=False,
               name=None,
               maximum_iterations=None,
               return_same_structure=False):
  """Repeat `body` while the condition `cond` is true.

  `cond` is a callable returning a boolean scalar tensor. `body` is a callable
  returning a (possibly nested) tuple, namedtuple or list of tensors of the same
  arity (length and structure) and types as `loop_vars`. `loop_vars` is a
  (possibly nested) tuple, namedtuple or list of tensors that is passed to both
  `cond` and `body`. `cond` and `body` both take as many arguments as there are
  `loop_vars`.

  In addition to regular Tensors or IndexedSlices, the body may accept and
  return TensorArray objects.  The flows of the TensorArray objects will
  be appropriately forwarded between loops and during gradient calculations.

  Note that `while_loop` calls `cond` and `body` *exactly once* (inside the
  call to `while_loop`, and not at all during `Session.run()`). `while_loop`
  stitches together the graph fragments created during the `cond` and `body`
  calls with some additional graph nodes to create the graph flow that
  repeats `body` until `cond` returns false.

  For correctness, `tf.while_loop()` strictly enforces shape invariants for
  the loop variables. A shape invariant is a (possibly partial) shape that
  is unchanged across the iterations of the loop. An error will be raised
  if the shape of a loop variable after an iteration is determined to be more
  general than or incompatible with its shape invariant. For example, a shape
  of [11, None] is more general than a shape of [11, 17], and [11, 21] is not
  compatible with [11, 17]. By default (if the argument `shape_invariants` is
  not specified), it is assumed that the initial shape of each tensor in
  `loop_vars` is the same in every iteration. The `shape_invariants` argument
  allows the caller to specify a less specific shape invariant for each loop
  variable, which is needed if the shape varies between iterations. The
  `tf.Tensor.set_shape`
  function may also be used in the `body` function to indicate that
  the output loop variable has a particular shape. The shape invariant for
  SparseTensor and IndexedSlices are treated specially as follows:

  a) If a loop variable is a SparseTensor, the shape invariant must be
  TensorShape([r]) where r is the rank of the dense tensor represented
  by the sparse tensor. It means the shapes of the three tensors of the
  SparseTensor are ([None], [None, r], [r]). NOTE: The shape invariant here
  is the shape of the SparseTensor.dense_shape property. It must be the shape of
  a vector.

  b) If a loop variable is an IndexedSlices, the shape invariant must be
  a shape invariant of the values tensor of the IndexedSlices. It means
  the shapes of the three tensors of the IndexedSlices are (shape, [shape[0]],
  [shape.ndims]).

  `while_loop` implements non-strict semantics, enabling multiple iterations
  to run in parallel. The maximum number of parallel iterations can be
  controlled by `parallel_iterations`, which gives users some control over
  memory consumption and execution order. For correct programs, `while_loop`
  should return the same result for any parallel_iterations > 0.

  For training, TensorFlow stores the tensors that are produced in the
  forward inference and are needed in back propagation. These tensors are a
  main source of memory consumption and often cause OOM errors when training
  on GPUs. When the flag swap_memory is true, we swap out these tensors from
  GPU to CPU. This for example allows us to train RNN models with very long
  sequences and large batches.

  Args:
    cond: A callable that represents the termination condition of the loop.
    body: A callable that represents the loop body.
    loop_vars: A (possibly nested) tuple, namedtuple or list of numpy array,
      `Tensor`, and `TensorArray` objects.
    shape_invariants: The shape invariants for the loop variables.
    parallel_iterations: The number of iterations allowed to run in parallel. It
      must be a positive integer.
    back_prop: Whether backprop is enabled for this while loop.
    swap_memory: Whether GPU-CPU memory swap is enabled for this loop.
    name: Optional name prefix for the returned tensors.
    maximum_iterations: Optional maximum number of iterations of the while loop
      to run.  If provided, the `cond` output is AND-ed with an additional
      condition ensuring the number of iterations executed is no greater than
      `maximum_iterations`.
    return_same_structure: If True, output has same structure as `loop_vars`. If
      eager execution is enabled, this is ignored (and always treated as True).

  Returns:
    The output tensors for the loop variables after the loop.
     If `return_same_structure` is True, the return value has the same
     structure as `loop_vars`.
     If `return_same_structure` is False, the return value is a Tensor,
     TensorArray or IndexedSlice if the length of `loop_vars` is 1, or a list
     otherwise.

  Raises:
    TypeError: if `cond` or `body` is not callable.
    ValueError: if `loop_vars` is empty.

  Example:

  ```python
  i = tf.constant(0)
  c = lambda i: tf.less(i, 10)
  b = lambda i: tf.add(i, 1)
  r = tf.while_loop(c, b, [i])
  ```

  Example with nesting and a namedtuple:

  ```python
  import collections
  Pair = collections.namedtuple('Pair', 'j, k')
  ijk_0 = (tf.constant(0), Pair(tf.constant(1), tf.constant(2)))
  c = lambda i, p: i < 10
  b = lambda i, p: (i + 1, Pair((p.j + p.k), (p.j - p.k)))
  ijk_final = tf.while_loop(c, b, ijk_0)
  ```

  Example using shape_invariants:

  ```python
  i0 = tf.constant(0)
  m0 = tf.ones([2, 2])
  c = lambda i, m: i < 10
  b = lambda i, m: [i+1, tf.concat([m, m], axis=0)]
  tf.while_loop(
      c, b, loop_vars=[i0, m0],
      shape_invariants=[i0.get_shape(), tf.TensorShape([None, 2])])
  ```

  Example which demonstrates non-strict semantics: In the following
  example, the final value of the counter `i` does not depend on `x`. So
  the `while_loop` can increment the counter parallel to updates of `x`.
  However, because the loop counter at one loop iteration depends
  on the value at the previous iteration, the loop counter itself cannot
  be incremented in parallel. Hence if we just want the final value of the
  counter (which we print on the line `print(sess.run(i))`), then
  `x` will never be incremented, but the counter will be updated on a
  single thread. Conversely, if we want the value of the output (which we
  print on the line `print(sess.run(out).shape)`), then the counter may be
  incremented on its own thread, while `x` can be incremented in
  parallel on a separate thread. In the extreme case, it is conceivable
  that the thread incrementing the counter runs until completion before
  `x` is incremented even a single time. The only thing that can never
  happen is that the thread updating `x` can never get ahead of the
  counter thread because the thread incrementing `x` depends on the value
  of the counter.

  ```python
  import tensorflow as tf

  n = 10000
  x = tf.constant(list(range(n)))
  c = lambda i, x: i < n
  b = lambda i, x: (tf.compat.v1.Print(i + 1, [i]), tf.compat.v1.Print(x + 1,
  [i], "x:"))
  i, out = tf.while_loop(c, b, (0, x))
  with tf.compat.v1.Session() as sess:
      print(sess.run(i))  # prints [0] ... [9999]

      # The following line may increment the counter and x in parallel.
      # The counter thread may get ahead of the other thread, but not the
      # other way around. So you may see things like
      # [9996] x:[9987]
      # meaning that the counter thread is on iteration 9996,
      # while the other thread is on iteration 9987
      print(sess.run(out).shape)
  ```
  """
  if not callable(cond):
    raise TypeError("'cond' must be callable.")
  if not callable(body):
    raise TypeError("'body' must be callable.")
  if parallel_iterations < 1:
    raise TypeError("'parallel_iterations' must be a positive integer.")

  loop_vars = variable_utils.convert_variables_to_tensors(loop_vars)

  # Always enable control flow v2 if building a function, regardless of toggle.
  executing_eagerly = context.executing_eagerly()
  if (util.EnableControlFlowV2(ops.get_default_graph()) and
      not executing_eagerly):
    return while_v2.while_loop(
        cond,
        body,
        loop_vars,
        shape_invariants=shape_invariants,
        parallel_iterations=parallel_iterations,
        maximum_iterations=maximum_iterations,
        name=name,
        return_same_structure=return_same_structure,
        back_prop=back_prop)

  with ops.name_scope(name, "while", loop_vars):
    if not loop_vars:
      raise ValueError("'loop_vars' must be provided.")
    try_to_pack = (len(loop_vars) == 1 and not return_same_structure)
    if maximum_iterations is not None:
      maximum_iterations = ops.convert_to_tensor(
          maximum_iterations, name="maximum_iterations")
      if maximum_iterations.shape.ndims != 0:
        raise ValueError("'maximum_iterations' must be a scalar. "
                         f"Received shape: {maximum_iterations.shape}")

      if executing_eagerly:
        counter = 0
        maximum_iterations = int(maximum_iterations.numpy())
      else:
        counter = constant_op.constant(
            0, dtype=maximum_iterations.dtype, name="iteration_counter")
      orig_cond = cond
      orig_body = body
      if try_to_pack:
        loop_vars = (counter, loop_vars[0])
        cond = lambda i, lv: (  # pylint: disable=g-long-lambda
            math_ops.logical_and(i < maximum_iterations, orig_cond(lv)))
        body = lambda i, lv: (i + 1, orig_body(lv))
      else:
        loop_vars = (counter, loop_vars)
        cond = lambda i, lv: (  # pylint: disable=g-long-lambda
            math_ops.logical_and(i < maximum_iterations, orig_cond(*lv)))
        body = lambda i, lv: (i + 1, orig_body(*lv))
      try_to_pack = False

    if executing_eagerly:
      packed = False  # whether the body result was packed into a 1-item tuple

      loop_var_structure = nest.map_structure(type_spec.type_spec_from_value,
                                              list(loop_vars))
      while cond(*loop_vars):
        loop_vars = body(*loop_vars)
        if try_to_pack and not isinstance(loop_vars, (list, tuple)):
          packed = True
          loop_vars = (loop_vars,)
        nest.assert_same_structure(loop_var_structure, list(loop_vars))

      def convert(x):
        if isinstance(x, tensor_array_ops.TensorArray):
          return x
        return ops.convert_to_tensor(x)

      loop_vars = nest.map_structure(convert, loop_vars, expand_composites=True)
      if maximum_iterations is not None:
        return loop_vars[1]
      else:
        return loop_vars[0] if packed else loop_vars

    if shape_invariants is not None:
      if maximum_iterations is not None:
        shape_invariants = (tensor_shape.TensorShape([]), shape_invariants)

    loop_context = control_flow_ops.WhileContext(
        maximum_iterations=maximum_iterations,
        parallel_iterations=parallel_iterations,
        back_prop=back_prop,
        swap_memory=swap_memory)
    # Only add non-nested loops to the collection. Any nested control flow will
    # be encapsulated in the root context.
    if loop_context.outer_context is None:
      ops.add_to_collection(ops.GraphKeys.WHILE_CONTEXT, loop_context)
    result = loop_context.BuildLoop(cond, body, loop_vars, shape_invariants,
                                    return_same_structure)
    if maximum_iterations is not None:
      return result[1]
    else:
      return result
