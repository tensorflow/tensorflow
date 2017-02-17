<!-- This file is machine generated: DO NOT EDIT! -->

# Higher Order Functions

Note: Functions taking `Tensor` arguments can also take anything accepted by
[`tf.convert_to_tensor`](framework.md#convert_to_tensor).

[TOC]

Functional operations. See the @{$python/functional_ops} guide.

- - -

### `tf.map_fn(fn, elems, dtype=None, parallel_iterations=10, back_prop=True, swap_memory=False, infer_shape=True, name=None)` {#map_fn}

map on the list of tensors unpacked from `elems` on dimension 0.

The simplest version of `map` repeatedly applies the callable `fn` to a
sequence of elements from first to last. The elements are made of the
tensors unpacked from `elems`. `dtype` is the data type of the return
value of `fn`. Users must provide `dtype` if it is different from
the data type of `elems`.

Suppose that `elems` is unpacked into `values`, a list of tensors. The shape
of the result tensor is `[values.shape[0]] + fn(values[0]).shape`.

This method also allows multi-arity `elems` and output of `fn`.  If `elems`
is a (possibly nested) list or tuple of tensors, then each of these tensors
must have a matching first (unpack) dimension.  The signature of `fn` may
match the structure of `elems`.  That is, if `elems` is
`(t1, [t2, t3, [t4, t5]])`, then an appropriate signature for `fn` is:
`fn = lambda (t1, [t2, t3, [t4, t5]]):`.

Furthermore, `fn` may emit a different structure than its input.  For example,
`fn` may look like: `fn = lambda t1: return (t1 + 1, t1 - 1)`.  In this case,
the `dtype` parameter is not optional: `dtype` must be a type or (possibly
nested) tuple of types matching the output of `fn`.

To apply a functional operation to the nonzero elements of a SparseTensor
one of the following methods is recommended. First, if the function is
expressible as TensorFlow ops, use

```python
  result = SparseTensor(input.indices, fn(input.values), input.dense_shape)
```

If, however, the function is not expressible as a TensorFlow op, then use

```python
result = SparseTensor(
  input.indices, map_fn(fn, input.values), input.dense_shape)
```

instead.

##### Args:


*  <b>`fn`</b>: The callable to be performed.  It accepts one argument, which will
    have the same (possibly nested) structure as `elems`.  Its output
    must have the same structure as `dtype` if one is provided, otherwise
    it must have the same structure as `elems`.
*  <b>`elems`</b>: A tensor or (possibly nested) sequence of tensors, each of which
    will be unpacked along their first dimension.  The nested sequence
    of the resulting slices will be applied to `fn`.
*  <b>`dtype`</b>: (optional) The output type(s) of `fn`.  If `fn` returns a structure
    of Tensors differing from the structure of `elems`, then `dtype` is not
    optional and must have the same structure as the output of `fn`.
*  <b>`parallel_iterations`</b>: (optional) The number of iterations allowed to run
    in parallel.
*  <b>`back_prop`</b>: (optional) True enables support for back propagation.
*  <b>`swap_memory`</b>: (optional) True enables GPU-CPU memory swapping.
*  <b>`infer_shape`</b>: (optional) False disables tests for consistent output shapes.
*  <b>`name`</b>: (optional) Name prefix for the returned tensors.

##### Returns:

  A tensor or (possibly nested) sequence of tensors.  Each tensor packs the
  results of applying `fn` to tensors unpacked from `elems` along the first
  dimension, from first to last.

##### Raises:


*  <b>`TypeError`</b>: if `fn` is not callable or the structure of the output of
    `fn` and `dtype` do not match, or if elems is a SparseTensor.
*  <b>`ValueError`</b>: if the lengths of the output of `fn` and `dtype` do not match.

##### Examples:

  ```python
  elems = np.array([1, 2, 3, 4, 5, 6])
  squares = map_fn(lambda x: x * x, elems)
  # squares == [1, 4, 9, 16, 25, 36]
  ```

  ```python
  elems = (np.array([1, 2, 3]), np.array([-1, 1, -1]))
  alternate = map_fn(lambda x: x[0] * x[1], elems, dtype=tf.int64)
  # alternate == [-1, 2, -3]
  ```

  ```python
  elems = np.array([1, 2, 3])
  alternates = map_fn(lambda x: (x, -x), elems, dtype=(tf.int64, tf.int64))
  # alternates[0] == [1, 2, 3]
  # alternates[1] == [-1, -2, -3]
  ```


- - -

### `tf.foldl(fn, elems, initializer=None, parallel_iterations=10, back_prop=True, swap_memory=False, name=None)` {#foldl}

foldl on the list of tensors unpacked from `elems` on dimension 0.

This foldl operator repeatedly applies the callable `fn` to a sequence
of elements from first to last. The elements are made of the tensors
unpacked from `elems` on dimension 0. The callable fn takes two tensors as
arguments. The first argument is the accumulated value computed from the
preceding invocation of fn. If `initializer` is None, `elems` must contain
at least one element, and its first element is used as the initializer.

Suppose that `elems` is unpacked into `values`, a list of tensors. The shape
of the result tensor is fn(initializer, values[0]).shape`.

##### Args:


*  <b>`fn`</b>: The callable to be performed.
*  <b>`elems`</b>: A tensor to be unpacked on dimension 0.
*  <b>`initializer`</b>: (optional) The initial value for the accumulator.
*  <b>`parallel_iterations`</b>: (optional) The number of iterations allowed to run
    in parallel.
*  <b>`back_prop`</b>: (optional) True enables support for back propagation.
*  <b>`swap_memory`</b>: (optional) True enables GPU-CPU memory swapping.
*  <b>`name`</b>: (optional) Name prefix for the returned tensors.

##### Returns:

  A tensor resulting from applying `fn` consecutively to the list of tensors
  unpacked from `elems`, from first to last.

##### Raises:


*  <b>`TypeError`</b>: if `fn` is not callable.

##### Example:

  ```python
  elems = [1, 2, 3, 4, 5, 6]
  sum = foldl(lambda a, x: a + x, elems)
  # sum == 21
  ```


- - -

### `tf.foldr(fn, elems, initializer=None, parallel_iterations=10, back_prop=True, swap_memory=False, name=None)` {#foldr}

foldr on the list of tensors unpacked from `elems` on dimension 0.

This foldr operator repeatedly applies the callable `fn` to a sequence
of elements from last to first. The elements are made of the tensors
unpacked from `elems`. The callable fn takes two tensors as arguments.
The first argument is the accumulated value computed from the preceding
invocation of fn. If `initializer` is None, `elems` must contain at least
one element, and its first element is used as the initializer.

Suppose that `elems` is unpacked into `values`, a list of tensors. The shape
of the result tensor is `fn(initializer, values[0]).shape`.

##### Args:


*  <b>`fn`</b>: The callable to be performed.
*  <b>`elems`</b>: A tensor that is unpacked into a sequence of tensors to apply `fn`.
*  <b>`initializer`</b>: (optional) The initial value for the accumulator.
*  <b>`parallel_iterations`</b>: (optional) The number of iterations allowed to run
    in parallel.
*  <b>`back_prop`</b>: (optional) True enables support for back propagation.
*  <b>`swap_memory`</b>: (optional) True enables GPU-CPU memory swapping.
*  <b>`name`</b>: (optional) Name prefix for the returned tensors.

##### Returns:

  A tensor resulting from applying `fn` consecutively to the list of tensors
  unpacked from `elems`, from last to first.

##### Raises:


*  <b>`TypeError`</b>: if `fn` is not callable.

##### Example:

  ```python
  elems = [1, 2, 3, 4, 5, 6]
  sum = foldr(lambda a, x: a + x, elems)
  # sum == 21
  ```


- - -

### `tf.scan(fn, elems, initializer=None, parallel_iterations=10, back_prop=True, swap_memory=False, infer_shape=True, name=None)` {#scan}

scan on the list of tensors unpacked from `elems` on dimension 0.

The simplest version of `scan` repeatedly applies the callable `fn` to a
sequence of elements from first to last. The elements are made of the tensors
unpacked from `elems` on dimension 0. The callable fn takes two tensors as
arguments. The first argument is the accumulated value computed from the
preceding invocation of fn. If `initializer` is None, `elems` must contain
at least one element, and its first element is used as the initializer.

Suppose that `elems` is unpacked into `values`, a list of tensors. The shape
of the result tensor is `[len(values)] + fn(initializer, values[0]).shape`.

This method also allows multi-arity `elems` and accumulator.  If `elems`
is a (possibly nested) list or tuple of tensors, then each of these tensors
must have a matching first (unpack) dimension.  The second argument of
`fn` must match the structure of `elems`.

If no `initializer` is provided, the output structure and dtypes of `fn`
are assumed to be the same as its input; and in this case, the first
argument of `fn` must match the structure of `elems`.

If an `initializer` is provided, then the output of `fn` must have the same
structure as `initializer`; and the first argument of `fn` must match
this structure.

For example, if `elems` is `(t1, [t2, t3])` and `initializer` is
`[i1, i2]` then an appropriate signature for `fn` in `python2` is:
`fn = lambda (acc_p1, acc_p2), (t1 [t2, t3]):` and `fn` must return a list,
`[acc_n1, acc_n2]`.  An alternative correct signature for `fn`, and the
 one that works in `python3`, is:
`fn = lambda a, t:`, where `a` and `t` correspond to the input tuples.

##### Args:


*  <b>`fn`</b>: The callable to be performed.  It accepts two arguments.  The first
    will have the same structure as `initializer` if one is provided,
    otherwise it will have the same structure as `elems`.  The second
    will have the same (possibly nested) structure as `elems`.  Its output
    must have the same structure as `initializer` if one is provided,
    otherwise it must have the same structure as `elems`.
*  <b>`elems`</b>: A tensor or (possibly nested) sequence of tensors, each of which
    will be unpacked along their first dimension.  The nested sequence
    of the resulting slices will be the first argument to `fn`.
*  <b>`initializer`</b>: (optional) A tensor or (possibly nested) sequence of tensors,
    initial value for the accumulator, and the expected output type of `fn`.
*  <b>`parallel_iterations`</b>: (optional) The number of iterations allowed to run
    in parallel.
*  <b>`back_prop`</b>: (optional) True enables support for back propagation.
*  <b>`swap_memory`</b>: (optional) True enables GPU-CPU memory swapping.
*  <b>`infer_shape`</b>: (optional) False disables tests for consistent output shapes.
*  <b>`name`</b>: (optional) Name prefix for the returned tensors.

##### Returns:

  A tensor or (possibly nested) sequence of tensors.  Each tensor packs the
  results of applying `fn` to tensors unpacked from `elems` along the first
  dimension, and the previous accumulator value(s), from first to last.

##### Raises:


*  <b>`TypeError`</b>: if `fn` is not callable or the structure of the output of
    `fn` and `initializer` do not match.
*  <b>`ValueError`</b>: if the lengths of the output of `fn` and `initializer`
    do not match.

##### Examples:

  ```python
  elems = np.array([1, 2, 3, 4, 5, 6])
  sum = scan(lambda a, x: a + x, elems)
  # sum == [1, 3, 6, 10, 15, 21]
  ```

  ```python
  elems = np.array([1, 2, 3, 4, 5, 6])
  initializer = np.array(0)
  sum_one = scan(
      lambda a, x: x[0] - x[1] + a, (elems + 1, elems), initializer)
  # sum_one == [1, 2, 3, 4, 5, 6]
  ```

  ```python
  elems = np.array([1, 0, 0, 0, 0, 0])
  initializer = (np.array(0), np.array(1))
  fibonaccis = scan(lambda a, _: (a[1], a[0] + a[1]), elems, initializer)
  # fibonaccis == ([1, 1, 2, 3, 5, 8], [1, 2, 3, 5, 8, 13])
  ```


