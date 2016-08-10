### `tf.contrib.layers.stack(inputs, layer, stack_args, **kwargs)` {#stack}

Builds a stack of layers by applying layer repeatedly using stack_args.

`stack` allows you to repeatedly apply the same operation with different
arguments `stack_args[i]`. For each application of the layer, `stack` creates
a new scope appended with an increasing number. For example:

```python
  y = stack(x, fully_connected, [32, 64, 128], scope='fc')
  # It is equivalent to:

  x = fully_connected(x, 32, scope='fc/fc_1')
  x = fully_connected(x, 64, scope='fc/fc_2')
  y = fully_connected(x, 128, scope='fc/fc_3')
```

If the `scope` argument is not given in `kwargs`, it is set to
`layer.__name__`, or `layer.func.__name__` (for `functools.partial`
objects). If neither `__name__` nor `func.__name__` is available, the
layers are called with `scope='stack'`.

##### Args:


*  <b>`inputs`</b>: A `Tensor` suitable for layer.
*  <b>`layer`</b>: A layer with arguments `(inputs, *args, **kwargs)`
*  <b>`stack_args`</b>: A list/tuple of parameters for each call of layer.
*  <b>`**kwargs`</b>: Extra kwargs for the layer.

##### Returns:

  a `Tensor` result of applying the stacked layers.

##### Raises:


*  <b>`ValueError`</b>: if the op is unknown or wrong.

