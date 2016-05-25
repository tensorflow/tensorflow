### `tf.map_fn(fn, elems, dtype=None, parallel_iterations=10, back_prop=True, swap_memory=False, name=None)` {#map_fn}

map on the list of tensors unpacked from `elems` on dimension 0.

This map operator repeatedly applies the callable `fn` to a sequence of
elements from first to last. The elements are made of the tensors unpacked
from `elems`. `dtype` is the data type of the return value of `fn`. Users
must provide `dtype` if it is different from the data type of `elems`.

Suppose that `elems` is unpacked into `values`, a list of tensors. The shape
of the result tensor is `[len(values)] + fn(values[0]).shape`.

##### Args:


*  <b>`fn`</b>: The callable to be performed.
*  <b>`elems`</b>: A tensor to be unpacked to apply `fn`.
*  <b>`dtype`</b>: (optional) The output type of `fn`.
*  <b>`parallel_iterations`</b>: (optional) The number of iterations allowed to run
                       in parallel.
*  <b>`back_prop`</b>: (optional) True enables back propagation.
*  <b>`swap_memory`</b>: (optional) True enables GPU-CPU memory swapping.
*  <b>`name`</b>: (optional) Name prefix for the returned tensors.

##### Returns:

  A tensor that packs the results of applying `fn` to the list of tensors
  unpacked from `elems`, from first to last.

##### Raises:


*  <b>`TypeError`</b>: if `fn` is not callable.

##### Example:

  ```python
  elems = [1, 2, 3, 4, 5, 6]
  squares = map_fn(lambda x: x * x, elems)
  # squares == [1, 4, 9, 16, 25, 36]
  ```

