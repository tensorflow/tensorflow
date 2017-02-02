<!-- This file is machine generated: DO NOT EDIT! -->

# Control Flow

Note: Functions taking `Tensor` arguments can also take anything accepted by
[`tf.convert_to_tensor`](framework.md#convert_to_tensor).

[TOC]

## Control Flow Operations

TensorFlow provides several operations and classes that you can use to control
the execution of operations and add conditional dependencies to your graph.

- - -

### `tf.identity(input, name=None)` {#identity}

Return a tensor with the same shape and contents as the input tensor or value.

##### Args:


*  <b>`input`</b>: A `Tensor`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `input`.


- - -

### `tf.tuple(tensors, name=None, control_inputs=None)` {#tuple}

Group tensors together.

This creates a tuple of tensors with the same values as the `tensors`
argument, except that the value of each tensor is only returned after the
values of all tensors have been computed.

`control_inputs` contains additional ops that have to finish before this op
finishes, but whose outputs are not returned.

This can be used as a "join" mechanism for parallel computations: all the
argument tensors can be computed in parallel, but the values of any tensor
returned by `tuple` are only available after all the parallel computations
are done.

See also `group` and `with_dependencies`.

##### Args:


*  <b>`tensors`</b>: A list of `Tensor`s or `IndexedSlices`, some entries can be `None`.
*  <b>`name`</b>: (optional) A name to use as a `name_scope` for the operation.
*  <b>`control_inputs`</b>: List of additional ops to finish before returning.

##### Returns:

  Same as `tensors`.

##### Raises:


*  <b>`ValueError`</b>: If `tensors` does not contain any `Tensor` or `IndexedSlices`.
*  <b>`TypeError`</b>: If `control_inputs` is not a list of `Operation` or `Tensor`
    objects.


- - -

### `tf.group(*inputs, **kwargs)` {#group}

Create an op that groups multiple operations.

When this op finishes, all ops in `input` have finished. This op has no
output.

See also `tuple` and `with_dependencies`.

##### Args:


*  <b>`*inputs`</b>: Zero or more tensors to group.
*  <b>`**kwargs`</b>: Optional parameters to pass when constructing the NodeDef.
*  <b>`name`</b>: A name for this operation (optional).

##### Returns:

  An Operation that executes all its inputs.

##### Raises:


*  <b>`ValueError`</b>: If an unknown keyword argument is provided.


- - -

### `tf.no_op(name=None)` {#no_op}

Does nothing. Only useful as a placeholder for control edges.

##### Args:


*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The created Operation.


- - -

### `tf.count_up_to(ref, limit, name=None)` {#count_up_to}

Increments 'ref' until it reaches 'limit'.

##### Args:


*  <b>`ref`</b>: A mutable `Tensor`. Must be one of the following types: `int32`, `int64`.
    Should be from a scalar `Variable` node.
*  <b>`limit`</b>: An `int`.
    If incrementing ref would bring it above limit, instead generates an
    'OutOfRange' error.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `ref`.
  A copy of the input before increment. If nothing else modifies the
  input, the values produced will all be distinct.


- - -

### `tf.cond(pred, fn1, fn2, name=None)` {#cond}

Return either fn1() or fn2() based on the boolean predicate `pred`.

`fn1` and `fn2` both return lists of output tensors. `fn1` and `fn2` must have
the same non-zero number and type of outputs.

Note that the conditional execution applies only to the operations defined in
fn1 and fn2. Consider the following simple program:

```python
z = tf.multiply(a, b)
result = tf.cond(x < y, lambda: tf.add(x, z), lambda: tf.square(y))
```

If x < y, the `tf.add` operation will be executed and `tf.square`
operation will not be executed. Since z is needed for at least one
branch of the cond, the `tf.multiply` operation is always executed, unconditionally.
Although this behavior is consistent with the dataflow model of TensorFlow,
it has occasionally surprised some users who expected a lazier semantics.

##### Args:


*  <b>`pred`</b>: A scalar determining whether to return the result of `fn1` or `fn2`.
*  <b>`fn1`</b>: The callable to be performed if pred is true.
*  <b>`fn2`</b>: The callable to be performed if pref is false.
*  <b>`name`</b>: Optional name prefix for the returned tensors.

##### Returns:

  Tensors returned by the call to either `fn1` or `fn2`. If the callables
  return a singleton list, the element is extracted from the list.

##### Raises:


*  <b>`TypeError`</b>: if `fn1` or `fn2` is not callable.
*  <b>`ValueError`</b>: if `fn1` and `fn2` do not return the same number of tensors, or
              return tensors of different types.


*  <b>`Example`</b>: 

```python
  x = tf.constant(2)
  y = tf.constant(5)
  def f1(): return tf.multiply(x, 17)
  def f2(): return tf.add(y, 23)
  r = tf.cond(tf.less(x, y), f1, f2)
  # r is set to f1().
  # Operations in f2 (e.g., tf.add) are not executed.
```


- - -

### `tf.case(pred_fn_pairs, default, exclusive=False, name='case')` {#case}

Create a case operation.

The `pred_fn_pairs` parameter is a dict or list of pairs of size N.
Each pair contains a boolean scalar tensor and a python callable that
creates the tensors to be returned if the boolean evaluates to True.
`default` is a callable generating a list of tensors. All the callables
in `pred_fn_pairs` as well as `default` should return the same number
and types of tensors.

If `exclusive==True`, all predicates are evaluated, and an exception is
thrown if more than one of the predicates evaluates to `True`.
If `exclusive==False`, execution stops are the first predicate which
evaluates to True, and the tensors generated by the corresponding function
are returned immediately. If none of the predicates evaluate to True, this
operation returns the tensors generated by `default`.

Example 1:
  Pseudocode:
  ```
    if (x < y) return 17;
    else return 23;
  ```

  Expressions:
  ```
    f1 = lambda: tf.constant(17)
    f2 = lambda: tf.constant(23)
    r = case([(tf.less(x, y), f1)], default=f2)
  ```

Example 2:
  Pseudocode:
  ```
    if (x < y && x > z) raise OpError("Only one predicate may evaluate true");
    if (x < y) return 17;
    else if (x > z) return 23;
    else return -1;
  ```

  Expressions:
  ```
    x = tf.constant(0)
    y = tf.constant(1)
    z = tf.constant(2)
    def f1(): return tf.constant(17)
    def f2(): return tf.constant(23)
    def f3(): return tf.constant(-1)
    r = case({tf.less(x, y): f1, tf.greater(x, z): f2},
             default=f3, exclusive=True)
  ```

##### Args:


*  <b>`pred_fn_pairs`</b>: Dict or list of pairs of a boolean scalar tensor and a
                 callable which returns a list of tensors.
*  <b>`default`</b>: A callable that returns a list of tensors.
*  <b>`exclusive`</b>: True iff at most one predicate is allowed to evaluate to `True`.
*  <b>`name`</b>: A name for this operation (optional).

##### Returns:

  The tensors returned by the first pair whose predicate evaluated to True, or
  those returned by `default` if none does.

##### Raises:


*  <b>`TypeError`</b>: If `pred_fn_pairs` is not a list/dictionary.
*  <b>`TypeError`</b>: If `pred_fn_pairs` is a list but does not contain 2-tuples.
*  <b>`TypeError`</b>: If `fns[i]` is not callable for any i, or `default` is not
             callable.


- - -

### `tf.while_loop(cond, body, loop_vars, shape_invariants=None, parallel_iterations=10, back_prop=True, swap_memory=False, name=None)` {#while_loop}

Repeat `body` while the condition `cond` is true.

`cond` is a callable returning a boolean scalar tensor. `body` is a callable
returning a (possibly nested) tuple, namedtuple or list of tensors of the same
arity (length and structure) and types as `loop_vars`. `loop_vars` is a
(possibly nested) tuple, namedtuple or list of tensors that is passed to both
`cond` and `body`. `cond` and `body` both take as many arguments as there are
`loop_vars`.

While `cond` evaluates to true, `body` is executed.

In addition to regular Tensors or IndexedSlices, the body may accept and
return TensorArray objects.  The flows of the TensorArray objects will
be appropriately forwarded between loops and during gradient calculations.

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
[`Tensor.set_shape()`](../../api_docs/python/framework.md#Tensor.set_shape)
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

For training, TensorFlow remembers the tensors that are produced in the
forward inference but needed in back propagation. These tensors can be a
main source of memory consumption and often cause OOM problems when training
on GPUs.  When the flag swap_memory is true, we swap out these tensors from
GPU to CPU.  This for example allows us to train RNN models with very long
sequences and large batches.

##### Args:


*  <b>`cond`</b>: A callable that represents the termination condition of the loop.
*  <b>`body`</b>: A callable that represents the loop body.
*  <b>`loop_vars`</b>: A (possibly nested) tuple, namedtuple or list of numpy array,
    `Tensor`, and `TensorArray` objects.
*  <b>`shape_invariants`</b>: The shape invariants for the loop variables.
*  <b>`parallel_iterations`</b>: The number of iterations allowed to run in parallel.
    It must be a positive integer.
*  <b>`back_prop`</b>: Whether backprop is enabled for this while loop.
*  <b>`swap_memory`</b>: Whether GPU-CPU memory swap is enabled for this loop.
*  <b>`name`</b>: Optional name prefix for the returned tensors.

##### Returns:

  The output tensors for the loop variables after the loop. When the length
  of `loop_vars` is 1 this is a Tensor, TensorArray or IndexedSlice and when
  the length of `loop_vars` is greater than 1 it returns a list.

##### Raises:


*  <b>`TypeError`</b>: if `cond` or `body` is not callable.
*  <b>`ValueError`</b>: if `loop_vars` is empty.


*  <b>`Example`</b>: 

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



## Logical Operators

TensorFlow provides several operations that you can use to add logical operators
to your graph.

- - -

### `tf.logical_and(x, y, name=None)` {#logical_and}

Returns the truth value of x AND y element-wise.

*NOTE*: `LogicalAnd` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

##### Args:


*  <b>`x`</b>: A `Tensor` of type `bool`.
*  <b>`y`</b>: A `Tensor` of type `bool`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `bool`.


- - -

### `tf.logical_not(x, name=None)` {#logical_not}

Returns the truth value of NOT x element-wise.

##### Args:


*  <b>`x`</b>: A `Tensor` of type `bool`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `bool`.


- - -

### `tf.logical_or(x, y, name=None)` {#logical_or}

Returns the truth value of x OR y element-wise.

*NOTE*: `LogicalOr` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

##### Args:


*  <b>`x`</b>: A `Tensor` of type `bool`.
*  <b>`y`</b>: A `Tensor` of type `bool`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `bool`.


- - -

### `tf.logical_xor(x, y, name='LogicalXor')` {#logical_xor}

x ^ y = (x | y) & ~(x & y).



## Comparison Operators

TensorFlow provides several operations that you can use to add comparison
operators to your graph.

- - -

### `tf.equal(x, y, name=None)` {#equal}

Returns the truth value of (x == y) element-wise.

*NOTE*: `Equal` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

##### Args:


*  <b>`x`</b>: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, `complex64`, `quint8`, `qint8`, `qint32`, `string`, `bool`, `complex128`.
*  <b>`y`</b>: A `Tensor`. Must have the same type as `x`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `bool`.


- - -

### `tf.not_equal(x, y, name=None)` {#not_equal}

Returns the truth value of (x != y) element-wise.

*NOTE*: `NotEqual` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

##### Args:


*  <b>`x`</b>: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, `complex64`, `quint8`, `qint8`, `qint32`, `string`, `bool`, `complex128`.
*  <b>`y`</b>: A `Tensor`. Must have the same type as `x`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `bool`.


- - -

### `tf.less(x, y, name=None)` {#less}

Returns the truth value of (x < y) element-wise.

*NOTE*: `Less` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

##### Args:


*  <b>`x`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
*  <b>`y`</b>: A `Tensor`. Must have the same type as `x`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `bool`.


- - -

### `tf.less_equal(x, y, name=None)` {#less_equal}

Returns the truth value of (x <= y) element-wise.

*NOTE*: `LessEqual` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

##### Args:


*  <b>`x`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
*  <b>`y`</b>: A `Tensor`. Must have the same type as `x`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `bool`.


- - -

### `tf.greater(x, y, name=None)` {#greater}

Returns the truth value of (x > y) element-wise.

*NOTE*: `Greater` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

##### Args:


*  <b>`x`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
*  <b>`y`</b>: A `Tensor`. Must have the same type as `x`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `bool`.


- - -

### `tf.greater_equal(x, y, name=None)` {#greater_equal}

Returns the truth value of (x >= y) element-wise.

*NOTE*: `GreaterEqual` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

##### Args:


*  <b>`x`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
*  <b>`y`</b>: A `Tensor`. Must have the same type as `x`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `bool`.


- - -

### `tf.where(condition, x=None, y=None, name=None)` {#where}

Return the elements, either from `x` or `y`, depending on the `condition`.

If both `x` and `y` are None, then this operation returns the coordinates of
true elements of `condition`.  The coordinates are returned in a 2-D tensor
where the first dimension (rows) represents the number of true elements, and
the second dimension (columns) represents the coordinates of the true
elements. Keep in mind, the shape of the output tensor can vary depending on
how many true values there are in input. Indices are output in row-major
order.

If both non-None, `x` and `y` must have the same shape.
The `condition` tensor must be a scalar if `x` and `y` are scalar.
If `x` and `y` are vectors or higher rank, then `condition` must be either a
vector with size matching the first dimension of `x`, or must have the same
shape as `x`.

The `condition` tensor acts as a mask that chooses, based on the value at each
element, whether the corresponding element / row in the output should be taken
from `x` (if true) or `y` (if false).

If `condition` is a vector and `x` and `y` are higher rank matrices, then it
chooses which row (outer dimension) to copy from `x` and `y`. If `condition`
has the same shape as `x` and `y`, then it chooses which element to copy from
`x` and `y`.

##### Args:


*  <b>`condition`</b>: A `Tensor` of type `bool`
*  <b>`x`</b>: A Tensor which may have the same shape as `condition`. If `condition` is
    rank 1, `x` may have higher rank, but its first dimension must match the
    size of `condition`.
*  <b>`y`</b>: A `tensor` with the same shape and type as `x`.
*  <b>`name`</b>: A name of the operation (optional)

##### Returns:

  A `Tensor` with the same type and shape as `x`, `y` if they are non-None.
  A `Tensor` with shape `(num_true, dim_size(condition))`.

##### Raises:


*  <b>`ValueError`</b>: When exactly one of `x` or `y` is non-None.



## Debugging Operations

TensorFlow provides several operations that you can use to validate values and
debug your graph.

- - -

### `tf.is_finite(x, name=None)` {#is_finite}

Returns which elements of x are finite.

@compatibility(numpy)
Equivalent to np.isfinite
@end_compatibility

##### Args:


*  <b>`x`</b>: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `bool`.


- - -

### `tf.is_inf(x, name=None)` {#is_inf}

Returns which elements of x are Inf.

@compatibility(numpy)
Equivalent to np.isinf
@end_compatibility

##### Args:


*  <b>`x`</b>: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `bool`.


- - -

### `tf.is_nan(x, name=None)` {#is_nan}

Returns which elements of x are NaN.

@compatibility(numpy)
Equivalent to np.isnan
@end_compatibility

##### Args:


*  <b>`x`</b>: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `bool`.


- - -

### `tf.verify_tensor_all_finite(t, msg, name=None)` {#verify_tensor_all_finite}

Assert that the tensor does not contain any NaN's or Inf's.

##### Args:


*  <b>`t`</b>: Tensor to check.
*  <b>`msg`</b>: Message to log on failure.
*  <b>`name`</b>: A name for this operation (optional).

##### Returns:

  Same tensor as `t`.


- - -

### `tf.check_numerics(tensor, message, name=None)` {#check_numerics}

Checks a tensor for NaN and Inf values.

When run, reports an `InvalidArgument` error if `tensor` has any values
that are not a number (NaN) or infinity (Inf). Otherwise, passes `tensor` as-is.

##### Args:


*  <b>`tensor`</b>: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
*  <b>`message`</b>: A `string`. Prefix of the error message.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `tensor`.


- - -

### `tf.add_check_numerics_ops()` {#add_check_numerics_ops}

Connect a `check_numerics` to every floating point tensor.

`check_numerics` operations themselves are added for each `half`, `float`,
or `double` tensor in the graph. For all ops in the graph, the
`check_numerics` op for all of its (`half`, `float`, or `double`) inputs
is guaranteed to run before the `check_numerics` op on any of its outputs.

##### Returns:

  A `group` op depending on all `check_numerics` ops added.


- - -

### `tf.Assert(condition, data, summarize=None, name=None)` {#Assert}

Asserts that the given condition is true.

If `condition` evaluates to false, print the list of tensors in `data`.
`summarize` determines how many entries of the tensors to print.

NOTE: To ensure that Assert executes, one usually attaches a dependency:

```python
# Ensure maximum element of x is smaller or equal to 1
assert_op = tf.Assert(tf.less_equal(tf.reduce_max(x), 1.), [x])
with tf.control_dependencies([assert_op]):
  ... code using x ...
```

##### Args:


*  <b>`condition`</b>: The condition to evaluate.
*  <b>`data`</b>: The tensors to print out when condition is false.
*  <b>`summarize`</b>: Print this many entries of each tensor.
*  <b>`name`</b>: A name for this operation (optional).

##### Returns:


*  <b>`assert_op`</b>: An `Operation` that, when executed, raises a
  `tf.errors.InvalidArgumentError` if `condition` is not true.


- - -

### `tf.Print(input_, data, message=None, first_n=None, summarize=None, name=None)` {#Print}

Prints a list of tensors.

This is an identity op with the side effect of printing `data` when
evaluating.

##### Args:


*  <b>`input_`</b>: A tensor passed through this op.
*  <b>`data`</b>: A list of tensors to print out when op is evaluated.
*  <b>`message`</b>: A string, prefix of the error message.
*  <b>`first_n`</b>: Only log `first_n` number of times. Negative numbers log always;
           this is the default.
*  <b>`summarize`</b>: Only print this many entries of each tensor. If None, then a
             maximum of 3 elements are printed per input tensor.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  Same tensor as `input_`.


